# -*- coding: utf-8 -*-

# @Author: Xianjun Li
# @E-mail: xjli@mail.hnust.edu.cn
# @Date: 2025/12/15 下午8:51
import torch.nn as nn
import torch
from timm.layers import DropPath

import math
import numpy as np
from einops.layers.torch import Rearrange

from lxj_utils_sys import print_colored
from wfa_main.Model_Dataset.model_base import MHSA
import torch.nn.functional as F
from einops import rearrange


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid = np.arange(grid_size, dtype=np.float32)
    # print(grid.shape)

    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans=1, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(1, 1))

    def forward(self, x):
        x = self.proj(x)
        return x


class CausalCNN(nn.Module):
    def __init__(self, in_channels, mid_channel, kernel_size=5):
        super(CausalCNN, self).__init__()

        self.kernel_size = kernel_size

        self.conv1 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1, padding=kernel_size - 1)
        self.bn1 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1, padding=kernel_size - 1)
        self.bn2 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()

        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.dropout1 = nn.Dropout(0.1)

        self.conv3 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1, padding=kernel_size - 1)
        self.bn3 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv1d(in_channels, mid_channel, kernel_size=kernel_size, stride=1, padding=kernel_size - 1)
        self.bn4 = nn.BatchNorm1d(mid_channel, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU()

        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        x = x.squeeze(2)

        x = self.conv1(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)[:, :, :-(self.kernel_size - 1)]
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.transpose(1, 2)

        return x, None


class ConvBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(ConvBlock1d, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation,
                      padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, dilation=dilation,
                      padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
        self.last_relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.last_relu(out + res)


class LocalProfiling(nn.Module):
    """ Local Profiling module """

    def __init__(self, in_channels, mid_channel):
        super().__init__()

        self.dividing = nn.Sequential(
            Rearrange('b c (n p) -> (b n) c p', n=4),
        )
        self.combination = nn.Sequential(
            Rearrange('(b n) c p -> b c (n p)', n=4),
        )

        self.net = nn.Sequential(
            ConvBlock1d(in_channels=in_channels, out_channels=32, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=32, out_channels=64, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=64, out_channels=128, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=128, out_channels=mid_channel, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        x = x.squeeze(2)
        # 切分方式： 使用 einops.Rearrange 将输入序列平均硬切
        # 分为 4 个不重叠的片段，并且是随机转一下再切分
        if self.training:
            sliding_size = np.random.randint(0, 1 + x.shape[-1] // 4)
            x = torch.roll(x, shifts=sliding_size, dims=-1)
        else:
            sliding_size = 0

        x = self.dividing(x)
        x = self.net(x)
        x = self.combination(x)

        x = x.permute(0, 2, 1)
        return x, sliding_size


class LocalProfiling_overlap(nn.Module):
    """
    放弃了 einops 的无缝切分，改用 SlidingWindowSplit。
    它允许相邻的两个分块之间有一定的重叠率（overlap_ratio）。
    片段长度被固定为 1800。
    """

    def __init__(self, in_channels, mid_channel, overlap_ratio=0.0):
        super().__init__()
        print_colored(f"重叠率：{overlap_ratio}", "yellow")
        segment_len = 1800
        self.dividing = SlidingWindowSplit(segment_len=segment_len, overlap_ratio=overlap_ratio)
        self.net = nn.Sequential(
            ConvBlock1d(in_channels=in_channels, out_channels=32, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=32, out_channels=64, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=64, out_channels=128, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
            ConvBlock1d(in_channels=128, out_channels=mid_channel, kernel_size=7),
            nn.MaxPool1d(kernel_size=8, stride=4),
            nn.Dropout(p=0.1),
        )
        # self.mhsa = MHSA(embed_dim=embed_dim, num_heads=embed_dim // 4,
        #                  num_mhsa_layers= depth, dim_feedforward=embed_dim * 4,
        #                  atten_config={'name': 'base'}
        #                  )

    def forward(self, x):
        B = x.shape[0]
        x = x.squeeze(2)

        if self.training:
            sliding_size = np.random.randint(0, 1 + x.shape[-1] // 4)
            x = torch.roll(x, shifts=sliding_size, dims=-1)
        else:
            sliding_size = 0

        x = self.dividing(x)
        x = self.net(x)
        x = Rearrange('(b n) c p -> b c (n p)', b=B)(x)

        x = x.permute(0, 2, 1)
        return x, sliding_size

def compute_out_size(length):
    length = math.floor((length - 8) / 4 + 1)
    length = math.floor((length - 8) / 4 + 1)
    length = math.floor((length - 8) / 4 + 1)
    length = math.floor((length - 8) / 4 + 1)
    return length


class RomerModel_EM1(nn.Module):
    def __init__(self,
                 num_classes,
                 feature_dim,
                 num_tabs,
                 max_matrix_len=1800,
                 drop_path_rate=0.1,
                 embed_dim=256,
                 depth=4,
                 early_stage=False,
                 **kwargs):
        super().__init__()
        print_colored("模型 -- 特征矩阵长度：{}".format(max_matrix_len), "yellow")
        self.early_stage = early_stage
        self.num_tabss = num_tabs
        # 多标签不支持早阶段流量识别
        assert num_tabs == 1 or not early_stage

        # 先进行特征嵌入（映射）
        self.patch_embed = PatchEmbed(patch_size=(feature_dim, 1), in_chans=1, embed_dim=embed_dim)

        if num_tabs == 1 or not kwargs.get("is_slice", True):
            # 如果是采用单标签，采用因果卷积或者（默认卷积）
            assert max_matrix_len % 6 == 0
            self.hidden_dim = max_matrix_len // 6
            self.local_model = CausalCNN(in_channels=embed_dim, mid_channel=embed_dim)
        else:
            # 如果是多标签，采用常规卷积
            assert max_matrix_len % 4 == 0
            self.local_model = LocalProfiling_overlap(in_channels=embed_dim, mid_channel=embed_dim,
                                                      overlap_ratio=kwargs.get("overlap_ratio", 0.1))
            self.hidden_dim = self.local_model.dividing.calculate_n(36000) * compute_out_size(1800)
            # print(f"\n>>>>>>>>>>> 多Tab，预分配最大序列维度:{self.hidden_dim + 1} <<<<<<<<<<<<<<<<<\n")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.hidden_dim + 1, embed_dim),
                                      requires_grad=False)

        self.mhsa = MHSA(embed_dim, embed_dim // 4, depth, embed_dim * 4,
                         atten_config={'name': 'base'}
                         )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.droppaths = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
            for i in range(depth)])
        self.fc_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2] - 1, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self, x, idx=None):
        # 1. 如果是三维 (B, C, L)，则升维成 (B, 1, C, L)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # 2. 强制校验：如果现在不是四维，或者第二维不是 1，则报错
        assert x.dim() == 4 and x.size(1) == 1, \
            f"输入维度错误！期望形状为 (B, 1, C, L) 或 (B, C, L)，但得到的是 {list(x.shape)}"
        # embed patches
        x = self.patch_embed(x)

        # local feature
        x, sliding_size = self.local_model(x)

        # append cls token
        cls_token = self.cls_token  # + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # mhsa已经带了位置编码，根据实际长度动态截取
        x = self.mhsa(x, self.pos_embed[:, :x.size(1), :])
        x = self.fc_norm(x)
        x = x[:, 1:, :]

        if self.early_stage and self.training:
            # early-stage training
            return self.fc(x)
        else:
            if idx is None:
                # early-stage
                x = x.mean(dim=1)
                x = self.fc(x)
                return x
            else:
                aggregate_idx = torch.floor(idx / 6).long()
                batch_size = x.size(0)
                selected_positions = [x[i, :aggregate_idx[i] + 1] for i in range(batch_size)]
                x = torch.stack([pos.mean(dim=0) for pos in selected_positions])
                x = self.fc(x)
                return x

# class RomerModel_NoOverlap(nn.Module):
#     def __init__(self,
#                  num_classes,
#                  feature_dim,
#                  num_tabs,
#                  max_matrix_len=1800,
#                  drop_path_rate=0.1,
#                  embed_dim=256,
#                  depth=4,
#                  early_stage=False,
#                  **kwargs):
#         super().__init__()
#
#         self.early_stage = early_stage
#         self.num_tabss = num_tabs
#         # 多标签不支持早阶段流量识别
#         assert num_tabs == 1 or not early_stage
#
#         # 移除 patch_embed，直接使用原始特征
#         # self.patch_embed = PatchEmbed(patch_size=(feature_dim, 1), in_chans=1, embed_dim=embed_dim)
#
#         if num_tabs == 1 or not kwargs.get("is_slice", True):
#             # 如果是采用单标签，采用因果卷积或者（默认卷积）
#             assert max_matrix_len % 6 == 0
#             self.local_model = CausalCNN(in_channels=feature_dim, mid_channel=embed_dim)
#             self.hidden_dim = max_matrix_len // 6
#         else:
#             # 如果是多标签，采用常规卷积
#             assert max_matrix_len % 4 == 0
#             # 强制重叠率为 0
#             self.local_model = LocalProfiling_overlap(in_channels=feature_dim, mid_channel=embed_dim,
#                                                       overlap_ratio=0.0)
#             self.hidden_dim = self.local_model.dividing.calculate_n(max_matrix_len) * compute_out_size(1800)
#
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.hidden_dim, embed_dim), requires_grad=False)
#
#         # ============ 基于标签原型查询的交叉注意力网络 (交叉注意力部分) ============
#         # Step 3.2: 引入可学习的标签原型 (Learnable Label Prototypes)
#         # 初始化一个形状为 (1, C, d) 的可学习张量，作为各个标签的“探照灯”
#         self.label_prototypes = nn.Parameter(torch.zeros(1, num_classes, embed_dim))
#         self.pos_embed_cls = nn.Parameter(torch.zeros(1, num_classes, embed_dim), requires_grad=False)
#
#         # Step 3.3: 交叉注意力聚合 (Cross-Attention Aggregation)
#         # 使用多头注意力，其中 Q 来自标签原型，K, V 来自流量局部特征
#         # 设置 num_heads=8 作为默认配置，避免 embed_dim // 32 可能带来的整除问题
#         num_heads = 8
#         while embed_dim % num_heads != 0:
#             num_heads -= 1
#             if num_heads <= 1:
#                 num_heads = 1
#                 break
#
#         # 交叉注意力
#         self.droppath = DropPath(drop_path_rate)
#         self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
#         self.cross_attn_norm = nn.LayerNorm(embed_dim)
#
#         # ============ 标签间的自注意力交互 (自注意力部分) ============
#         # Step 4 & 5 融合：直接将 (B, C, d) 输入到 MHSA 捕捉共现关系
#         self.mhsa = MHSA(embed_dim, embed_dim // 4, depth, embed_dim * 4,
#                          atten_config={'name': 'base'}
#                          )
#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         self.droppaths = nn.ModuleList([
#             DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
#             for i in range(depth)])
#
#         self.fc_norm = nn.LayerNorm(embed_dim)
#
#         # ============ 多标签分类头 (分类头部分) ============
#         # Step 6：沿着 d 维度进行全连接映射
#         self.fc = nn.Linear(embed_dim, 1)
#
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         # 移除 patch_embed 初始化
#         # w = self.patch_embed.proj.weight.data
#         # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#
#         # 初始化 label prototypes
#         torch.nn.init.normal_(self.label_prototypes, std=0.02)
#
#         pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2], cls_token=False)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
#
#         pos_embed_cls = get_1d_sincos_pos_embed(self.pos_embed_cls.shape[-1], self.pos_embed_cls.shape[-2],
#                                                 cls_token=False)
#         self.pos_embed_cls.data.copy_(torch.from_numpy(pos_embed_cls).float().unsqueeze(0))
#
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def no_weight_decay(self):
#         return {'label_prototypes', 'pos_embed', 'pos_embed_cls'}
#
#     def forward(self, x, idx=None):
#         # 1. 如果是三维 (B, C, L)，则升维成 (B, 1, C, L)
#         if x.dim() == 3:
#             x = x.unsqueeze(1)
#         # 2. 强制校验：如果现在不是四维，或者第二维不是 1，则报错
#         assert x.dim() == 4 and x.size(1) == 1, \
#             f"输入维度错误！期望形状为 (B, 1, C, L) 或 (B, C, L)，但得到的是 {list(x.shape)}"
#
#         # 移除 patch_embed，直接使用原始特征
#         # x = self.patch_embed(x)
#         # 将 (B, 1, C, L) 转换为 (B, C, 1, L)，以便后续 local_model (squeeze(2)) 处理
#         x = x.transpose(1, 2)
#
#         # local feature
#         # Step 3.1: 构建全局特征证据池 (Constructing the Evidence Pool)
#         # 输出的 x 形状相当于 (B, k*L, d) -> 作为 Key 和 Value
#         x, sliding_size = self.local_model(x)
#
#         B = x.shape[0]
#         # Step 3.2: 生成 Query (B, C, d)
#         q = self.label_prototypes.expand(B, -1, -1)
#
#         # 将位置编码添加到 x (第一层交叉注意力开始前)
#         x = x + self.pos_embed[:, :x.size(1), :]
#         # Step 3.3: 交叉注意力聚合 (Cross-Attention Aggregation)
#         attn_out, _ = self.cross_attn(query=q, key=x, value=x)
#         # 加上残差连接并归一化，输出形状保持 (B, C, d)
#         x = q + self.droppath(self.cross_attn_norm(q + attn_out))
#
#         # # Step 4 & 5: 标签间的自注意力交互
#         # # x 形如 (B, C, d)
#         # x = x + self.pos_embed_cls
#         # x = self.mhsa(x)
#         # x = self.fc_norm(x)
#
#         # Step 6：多标签分类头
#         # 沿着 d 维度进行全连接映射，输出 logits
#         # (B, C, d) -> FC -> (B, C, 1) -> Squeeze -> (B, C)
#
#         if self.early_stage and self.training:
#             # early-stage training 不支持
#             return self.fc(x).squeeze(-1)
#         else:
#             if idx is None:
#                 # normal training / test
#                 x = self.fc(x).squeeze(-1)
#                 return x
#             else:
#                 # idx exists handling (不过多标签不支持 early stage，为了格式一致保留)
#                 x = self.fc(x).squeeze(-1)
#                 return x
#
#
# class RomerModel_NoAtt(nn.Module):
#     def __init__(self,
#                  num_classes,
#                  feature_dim,
#                  num_tabs,
#                  max_matrix_len=1800,
#                  drop_path_rate=0.1,
#                  embed_dim=256,
#                  depth=4,
#                  early_stage=False,
#                  **kwargs):
#         super().__init__()
#         print_colored("模型 -- 特征矩阵长度：{}".format(max_matrix_len), "yellow")
#         self.early_stage = early_stage
#         self.num_tabss = num_tabs
#         # 多标签不支持早阶段流量识别
#         assert num_tabs == 1 or not early_stage
#
#         # 先进行特征嵌入（映射）
#         self.patch_embed = PatchEmbed(patch_size=(feature_dim, 1), in_chans=1, embed_dim=embed_dim)
#
#         if num_tabs == 1 or not kwargs.get("is_slice", True):
#             # 如果是采用单标签，采用因果卷积或者（默认卷积）
#             assert max_matrix_len % 6 == 0
#             self.hidden_dim = max_matrix_len // 6
#             self.local_model = CausalCNN(in_channels=embed_dim, mid_channel=embed_dim)
#         else:
#             # 如果是多标签，采用常规卷积
#             assert max_matrix_len % 4 == 0
#             self.local_model = LocalProfiling_overlap(in_channels=embed_dim, mid_channel=embed_dim,
#                                                       overlap_ratio=kwargs.get("overlap_ratio", 0.1))
#             self.hidden_dim = self.local_model.dividing.calculate_n(36000) * compute_out_size(1800)
#             # print(f"\n>>>>>>>>>>> 多Tab，预分配最大序列维度:{self.hidden_dim + 1} <<<<<<<<<<<<<<<<<\n")
#
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, self.hidden_dim + 1, embed_dim),
#                                       requires_grad=False)
#
#         self.mhsa = MHSA(embed_dim, embed_dim // 4, depth, embed_dim * 4,
#                          atten_config={'name': 'base'}
#                          )
#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
#         self.droppaths = nn.ModuleList([
#             DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
#             for i in range(depth)])
#         self.fc_norm = nn.LayerNorm(embed_dim)
#         self.fc = nn.Linear(embed_dim, num_classes)
#         self.initialize_weights()
#
#     def initialize_weights(self):
#         pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2] - 1, cls_token=True)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
#
#         w = self.patch_embed.proj.weight.data
#         torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
#
#         torch.nn.init.normal_(self.cls_token, std=.02)
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#
#     def no_weight_decay(self):
#         return {'pos_embed', 'cls_token', 'dist_token'}
#
#     def forward(self, x, idx=None):
#         # 1. 如果是三维 (B, C, L)，则升维成 (B, 1, C, L)
#         if x.dim() == 3:
#             x = x.unsqueeze(1)
#         # 2. 强制校验：如果现在不是四维，或者第二维不是 1，则报错
#         assert x.dim() == 4 and x.size(1) == 1, \
#             f"输入维度错误！期望形状为 (B, 1, C, L) 或 (B, C, L)，但得到的是 {list(x.shape)}"
#         # embed patches
#         x = self.patch_embed(x)
#
#         # local feature
#         x, sliding_size = self.local_model(x)
#
#         ## 消融实验，也就是没有注意力机制
#         # # append cls token
#         # cls_token = self.cls_token  # + self.pos_embed[:, :1, :]
#         # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
#         # x = torch.cat((cls_tokens, x), dim=1)
#         #
#         # # mhsa已经带了位置编码，根据实际长度动态截取
#         # x = self.mhsa(x, self.pos_embed[:, :x.size(1), :])
#         # x = self.fc_norm(x)
#         # x = x[:, 1:, :]
#
#         if self.early_stage and self.training:
#             # early-stage training
#             return self.fc(x)
#         else:
#             if idx is None:
#                 # early-stage
#                 x = x.mean(dim=1)
#                 x = self.fc(x)
#                 return x
#             else:
#                 aggregate_idx = torch.floor(idx / 6).long()
#                 batch_size = x.size(0)
#                 selected_positions = [x[i, :aggregate_idx[i] + 1] for i in range(batch_size)]
#                 x = torch.stack([pos.mean(dim=0) for pos in selected_positions])
#                 x = self.fc(x)
#                 return x

class RomerModel_EM3(nn.Module):
    def __init__(self,
                 num_classes,
                 feature_dim,
                 num_tabs,
                 max_matrix_len=1800,
                 drop_path_rate=0.1,
                 embed_dim=256,
                 depth=4,
                 early_stage=False,
                 **kwargs):
        super().__init__()

        self.early_stage = early_stage
        self.num_tabss = num_tabs
        # 多标签不支持早阶段流量识别
        assert num_tabs == 1 or not early_stage

        # 先进行特征嵌入（映射）
        self.patch_embed = PatchEmbed(patch_size=(feature_dim, 1), in_chans=1, embed_dim=embed_dim)

        if num_tabs == 1 or not kwargs.get("is_slice", True):
            # 如果是采用单标签，采用因果卷积或者（默认卷积）
            assert max_matrix_len % 6 == 0
            self.local_model = CausalCNN(in_channels=embed_dim, mid_channel=embed_dim)
            self.hidden_dim = max_matrix_len // 6
        else:
            # 如果是多标签，采用常规卷积
            assert max_matrix_len % 4 == 0
            self.local_model = LocalProfiling_overlap(in_channels=embed_dim, mid_channel=embed_dim,
                                                      overlap_ratio=kwargs.get("overlap_ratio", 0.1))
            self.hidden_dim = self.local_model.dividing.calculate_n(36000) * compute_out_size(1800)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.hidden_dim, embed_dim), requires_grad=False)

        # ============ 基于标签原型查询的交叉注意力网络 (交叉注意力部分) ============
        # Step 3.2: 引入可学习的标签原型 (Learnable Label Prototypes)
        # 初始化一个形状为 (1, C, d) 的可学习张量，作为各个标签的“探照灯”
        self.label_prototypes = nn.Parameter(torch.zeros(1, num_classes, embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, num_classes, embed_dim), requires_grad=False)

        # Step 3.3: 交叉注意力聚合 (Cross-Attention Aggregation)
        # 使用多头注意力，其中 Q 来自标签原型，K, V 来自流量局部特征
        # 设置 num_heads=8 作为默认配置，避免 embed_dim // 32 可能带来的整除问题
        num_heads = 8
        while embed_dim % num_heads != 0:
            num_heads -= 1
            if num_heads <= 1:
                num_heads = 1
                break

        # 交叉注意力
        self.droppath = DropPath(drop_path_rate)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)

        # ============ 标签间的自注意力交互 (自注意力部分) ============
        # Step 4 & 5 融合：直接将 (B, C, d) 输入到 MHSA 捕捉共现关系
        self.mhsa = MHSA(embed_dim, embed_dim // 4, depth, embed_dim * 4,
                         atten_config={'name': 'base'}
                         )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.droppaths = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
            for i in range(depth)])

        self.fc_norm = nn.LayerNorm(embed_dim)

        # ============ 多标签分类头 (分类头部分) ============
        # Step 6：沿着 d 维度进行全连接映射
        self.fc = nn.Linear(embed_dim, 1)

        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # 初始化 label prototypes
        torch.nn.init.normal_(self.label_prototypes, std=0.02)

        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_embed_cls = get_1d_sincos_pos_embed(self.pos_embed_cls.shape[-1], self.pos_embed_cls.shape[-2],
                                                cls_token=False)
        self.pos_embed_cls.data.copy_(torch.from_numpy(pos_embed_cls).float().unsqueeze(0))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'label_prototypes', 'pos_embed', 'pos_embed_cls'}

    def forward(self, x, idx=None):
        # 1. 如果是三维 (B, C, L)，则升维成 (B, 1, C, L)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # 2. 强制校验：如果现在不是四维，或者第二维不是 1，则报错
        assert x.dim() == 4 and x.size(1) == 1, \
            f"输入维度错误！期望形状为 (B, 1, C, L) 或 (B, C, L)，但得到的是 {list(x.shape)}"

        # embed patches
        x = self.patch_embed(x)

        # local feature
        # Step 3.1: 构建全局特征证据池 (Constructing the Evidence Pool)
        # 输出的 x 形状相当于 (B, k*L, d) -> 作为 Key 和 Value
        x, sliding_size = self.local_model(x)

        B = x.shape[0]
        # Step 3.2: 生成 Query (B, C, d)
        q = self.label_prototypes.expand(B, -1, -1)

        # 将位置编码添加到 x (第一层交叉注意力开始前)
        x = x + self.pos_embed[:, :x.size(1), :]
        # Step 3.3: 交叉注意力聚合 (Cross-Attention Aggregation)
        attn_out, _ = self.cross_attn(query=q, key=x, value=x)
        # 加上残差连接并归一化，输出形状保持 (B, C, d)
        x = q + self.droppath(self.cross_attn_norm(q + attn_out))

        # Step 4 & 5: 标签间的自注意力交互
        # x 形如 (B, C, d)
        x = x + self.pos_embed_cls
        x = self.mhsa(x)
        x = self.fc_norm(x)

        # Step 6：多标签分类头
        # 沿着 d 维度进行全连接映射，输出 logits
        # (B, C, d) -> FC -> (B, C, 1) -> Squeeze -> (B, C)

        if self.early_stage and self.training:
            # early-stage training 不支持
            return self.fc(x).squeeze(-1)
        else:
            if idx is None:
                # normal training / test
                x = self.fc(x).squeeze(-1)
                return x
            else:
                # idx exists handling (不过多标签不支持 early stage，为了格式一致保留)
                x = self.fc(x).squeeze(-1)
                return x


class RomerModel_EM1_Overlap(nn.Module):
    def __init__(self,
                 num_classes,
                 feature_dim,
                 num_tabs,
                 max_matrix_len=1800,
                 drop_path_rate=0.1,
                 embed_dim=256,
                 depth=4,
                 early_stage=False,
                 **kwargs):
        super().__init__()
        print_colored("模型(EM1_Overlap) -- 特征矩阵长度：{}".format(max_matrix_len), "yellow")
        self.early_stage = early_stage
        self.num_tabss = num_tabs
        # 多标签不支持早阶段流量识别
        assert num_tabs == 1 or not early_stage

        # 取消特征嵌入
        # self.patch_embed = PatchEmbed(patch_size=(feature_dim, 1), in_chans=1, embed_dim=embed_dim)

        if num_tabs == 1 or not kwargs.get("is_slice", True):
            assert max_matrix_len % 6 == 0
            self.hidden_dim = max_matrix_len // 6
            self.local_model = CausalCNN(in_channels=feature_dim, mid_channel=embed_dim)
        else:
            assert max_matrix_len % 4 == 0
            # 设置 overlap 为 0
            self.local_model = LocalProfiling_overlap(in_channels=feature_dim, mid_channel=embed_dim,
                                                      overlap_ratio=0.0)
            self.hidden_dim = self.local_model.dividing.calculate_n(36000) * compute_out_size(1800)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.hidden_dim + 1, embed_dim),
                                      requires_grad=False)

        self.mhsa = MHSA(embed_dim, embed_dim // 4, depth, embed_dim * 4,
                         atten_config={'name': 'base'}
                         )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.droppaths = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
            for i in range(depth)])
        self.fc_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2] - 1, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def forward(self, x, idx=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 4 and x.size(1) == 1, \
            f"输入维度错误！期望形状为 (B, 1, C, L) 或 (B, C, L)，但得到的是 {list(x.shape)}"
        
        # 取消特征嵌入，直接转置以适配 local_model
        x = x.transpose(1, 2)

        # local feature
        x, sliding_size = self.local_model(x)

        # append cls token
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # mhsa已经带了位置编码
        x = self.mhsa(x, self.pos_embed[:, :x.size(1), :])
        x = self.fc_norm(x)
        x = x[:, 1:, :]

        if self.early_stage and self.training:
            return self.fc(x)
        else:
            if idx is None:
                x = x.mean(dim=1)
                x = self.fc(x)
                return x
            else:
                aggregate_idx = torch.floor(idx / 6).long()
                batch_size = x.size(0)
                selected_positions = [x[i, :aggregate_idx[i] + 1] for i in range(batch_size)]
                x = torch.stack([pos.mean(dim=0) for pos in selected_positions])
                x = self.fc(x)
                return x


class RomerModel_EM3_Overlap(nn.Module):
    def __init__(self,
                 num_classes,
                 feature_dim,
                 num_tabs,
                 max_matrix_len=1800,
                 drop_path_rate=0.1,
                 embed_dim=256,
                 depth=4,
                 early_stage=False,
                 **kwargs):
        super().__init__()

        self.early_stage = early_stage
        self.num_tabss = num_tabs
        assert num_tabs == 1 or not early_stage

        # 取消特征嵌入
        # self.patch_embed = PatchEmbed(patch_size=(feature_dim, 1), in_chans=1, embed_dim=embed_dim)

        if num_tabs == 1 or not kwargs.get("is_slice", True):
            assert max_matrix_len % 6 == 0
            self.local_model = CausalCNN(in_channels=feature_dim, mid_channel=embed_dim)
            self.hidden_dim = max_matrix_len // 6
        else:
            assert max_matrix_len % 4 == 0
            # 设置 overlap 为 0
            self.local_model = LocalProfiling_overlap(in_channels=feature_dim, mid_channel=embed_dim,
                                                      overlap_ratio=0.0)
            self.hidden_dim = self.local_model.dividing.calculate_n(36000) * compute_out_size(1800)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.hidden_dim, embed_dim), requires_grad=False)

        self.label_prototypes = nn.Parameter(torch.zeros(1, num_classes, embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, num_classes, embed_dim), requires_grad=False)

        num_heads = 8
        while embed_dim % num_heads != 0:
            num_heads -= 1
            if num_heads <= 1:
                num_heads = 1
                break

        self.droppath = DropPath(drop_path_rate)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(embed_dim)

        self.mhsa = MHSA(embed_dim, embed_dim // 4, depth, embed_dim * 4,
                         atten_config={'name': 'base'}
                         )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.droppaths = nn.ModuleList([
            DropPath(dpr[i]) if dpr[i] > 0.0 else nn.Identity()
            for i in range(depth)])

        self.fc_norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, 1)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.label_prototypes, std=0.02)

        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.pos_embed.shape[-2], cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        pos_embed_cls = get_1d_sincos_pos_embed(self.pos_embed_cls.shape[-1], self.pos_embed_cls.shape[-2],
                                                cls_token=False)
        self.pos_embed_cls.data.copy_(torch.from_numpy(pos_embed_cls).float().unsqueeze(0))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'label_prototypes', 'pos_embed', 'pos_embed_cls'}

    def forward(self, x, idx=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 4 and x.size(1) == 1, \
            f"输入维度错误！期望形状为 (B, 1, C, L) 或 (B, C, L)，但得到的是 {list(x.shape)}"

        # 取消特征嵌入，直接转置以适配 local_model
        x = x.transpose(1, 2)

        x, sliding_size = self.local_model(x)

        B = x.shape[0]
        q = self.label_prototypes.expand(B, -1, -1)

        x = x + self.pos_embed[:, :x.size(1), :]
        attn_out, _ = self.cross_attn(query=q, key=x, value=x)
        x = q + self.droppath(self.cross_attn_norm(q + attn_out))

        x = x + self.pos_embed_cls
        x = self.mhsa(x)
        x = self.fc_norm(x)

        if self.early_stage and self.training:
            return self.fc(x).squeeze(-1)
        else:
            x = self.fc(x).squeeze(-1)
            return x


class RomerModel_EM1_Atten(nn.Module):
    def __init__(self,
                 num_classes,
                 feature_dim,
                 num_tabs,
                 max_matrix_len=1800,
                 drop_path_rate=0.1,
                 embed_dim=256,
                 depth=4,
                 early_stage=False,
                 **kwargs):
        super().__init__()
        print_colored("模型(EM1_Atten) -- 特征矩阵长度：{}".format(max_matrix_len), "yellow")
        self.early_stage = early_stage
        self.num_tabss = num_tabs
        assert num_tabs == 1 or not early_stage

        self.patch_embed = PatchEmbed(patch_size=(feature_dim, 1), in_chans=1, embed_dim=embed_dim)

        if num_tabs == 1 or not kwargs.get("is_slice", True):
            assert max_matrix_len % 6 == 0
            self.hidden_dim = max_matrix_len // 6
            self.local_model = CausalCNN(in_channels=embed_dim, mid_channel=embed_dim)
        else:
            assert max_matrix_len % 4 == 0
            self.local_model = LocalProfiling_overlap(in_channels=embed_dim, mid_channel=embed_dim,
                                                      overlap_ratio=kwargs.get("overlap_ratio", 0.1))
            self.hidden_dim = self.local_model.dividing.calculate_n(36000) * compute_out_size(1800)

        # 去除 attention 部分: mhsa, pos_embed, cls_token 都移除
        self.fc = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, idx=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 4 and x.size(1) == 1, \
            f"输入维度错误！期望形状为 (B, 1, C, L) 或 (B, C, L)，但得到的是 {list(x.shape)}"
        
        x = self.patch_embed(x)

        # local feature
        x, sliding_size = self.local_model(x)

        # 没有 attention，直接 pooling
        if self.early_stage and self.training:
            return self.fc(x)
        else:
            if idx is None:
                x = x.mean(dim=1)
                x = self.fc(x)
                return x
            else:
                aggregate_idx = torch.floor(idx / 6).long()
                batch_size = x.size(0)
                selected_positions = [x[i, :aggregate_idx[i] + 1] for i in range(batch_size)]
                x = torch.stack([pos.mean(dim=0) for pos in selected_positions])
                x = self.fc(x)
                return x


class RomerModel_EM3_Atten(nn.Module):
    def __init__(self,
                 num_classes,
                 feature_dim,
                 num_tabs,
                 max_matrix_len=1800,
                 drop_path_rate=0.1,
                 embed_dim=256,
                 depth=4,
                 early_stage=False,
                 **kwargs):
        super().__init__()

        self.early_stage = early_stage
        self.num_tabss = num_tabs
        assert num_tabs == 1 or not early_stage

        self.patch_embed = PatchEmbed(patch_size=(feature_dim, 1), in_chans=1, embed_dim=embed_dim)

        if num_tabs == 1 or not kwargs.get("is_slice", True):
            assert max_matrix_len % 6 == 0
            self.local_model = CausalCNN(in_channels=embed_dim, mid_channel=embed_dim)
            self.hidden_dim = max_matrix_len // 6
        else:
            assert max_matrix_len % 4 == 0
            self.local_model = LocalProfiling_overlap(in_channels=embed_dim, mid_channel=embed_dim,
                                                      overlap_ratio=kwargs.get("overlap_ratio", 0.1))
            self.hidden_dim = self.local_model.dividing.calculate_n(36000) * compute_out_size(1800)

        # 去除 attention 部分: label_prototypes, pos_embed, cross_attn, mhsa 均移除
        # 直接通过全连接层输出 (B, num_classes)
        self.fc = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, idx=None):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        assert x.dim() == 4 and x.size(1) == 1, \
            f"输入维度错误！期望形状为 (B, 1, C, L) 或 (B, C, L)，但得到的是 {list(x.shape)}"

        x = self.patch_embed(x)

        # local feature
        x, sliding_size = self.local_model(x)

        # 没有 attention，直接 pooling
        if self.early_stage and self.training:
            return self.fc(x)
        else:
            if idx is None:
                x = x.mean(dim=1)
                x = self.fc(x)
                return x
            else:
                x = x.mean(dim=1)
                x = self.fc(x)
                return x


class SlidingWindowSplit(nn.Module):
    def __init__(self, segment_len, overlap_ratio=0.0, padding_value=0):
        super().__init__()
        self.segment_len = segment_len
        self.padding_value = padding_value
        # 步长计算
        self.stride = int(segment_len * (1 - overlap_ratio))
        if self.stride < 1:
            self.stride = 1

    def calculate_n(self, l):
        """
        根据输入长度 l，计算分段后的窗口数量 n
        """
        if l < self.segment_len:
            # 长度不足一个片段，padding 到 segment_len，窗口数为 1
            return 1

        # 计算 padding 后的长度 (逻辑同 forward)
        remainder = (l - self.segment_len) % self.stride
        pad_len = 0 if remainder == 0 else self.stride - remainder
        l_pad = l + pad_len

        # 计算窗口数 n: (L_pad - segment_len) / stride + 1
        n = (l_pad - self.segment_len) // self.stride + 1
        return n

    def forward(self, x):
        b, c, l = x.shape

        # 1. 自动计算 padding
        if l < self.segment_len:
            pad_len = self.segment_len - l
        else:
            remainder = (l - self.segment_len) % self.stride
            pad_len = 0 if remainder == 0 else self.stride - remainder

        if pad_len > 0:
            x = F.pad(x, (0, pad_len), value=self.padding_value)

        # 2. 滑动窗口拆分
        # 此时的 x 维度: (b, c, l_pad)
        x = x.unfold(dimension=-1, size=self.segment_len, step=self.stride)

        # 3. 获取窗口数 n (x 维度变为 b, c, n, p)
        n = x.shape[2]

        # 验证逻辑：确保 calculate_n 的结果和实际 unfold 的维度一致
        # assert n == self.calculate_n(l)

        x = rearrange(x, 'b c n p -> (b n) c p')
        return x


class GateNet(nn.Module):
    def __init__(self):
        super(GateNet, self).__init__()
        self.feature_dim = 512
        # RF_model 现在会返回确定的特征维度
        self.model = RF_model(out_feature_dim=self.feature_dim)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.gating_fc = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.model(x)
        x = self.gap(feat)
        x = x.view(x.size(0), -1)
        h = self.gating_fc(x)
        return h


class RF_model(nn.Module):
    def __init__(self, out_feature_dim=512):
        super(RF_model, self).__init__()
        self.first_layer = make_first_layers()

        # 核心修正：make_first_layers 最终输出的是 64 个通道
        # 如果经过 view 转换，我们需要确保 1D 卷积的输入通道匹配
        self.features = make_layers([128, 128, 'M', 256, 256, 'M', 512, out_feature_dim], in_channels=64)

        self._initialize_weights()

    def forward(self, x):
        # 1. 经过 2D 卷积层: input [B, 1, 5, 1800] -> output [B, 64, H', W']
        x = self.first_layer(x)

        # 2. 核心修正：处理维度以适配 1D 卷积
        # 这里的 H' 因为卷积和池化会变小。报错显示你的 H' 此时被 view 压进了通道。
        # 我们使用 mean 或者是 view 重新排列，确保通道数依然是 64
        # 假设我们只关心时间维度上的特征映射：
        x = torch.mean(x, dim=2)  # 将高度维度 H' 压缩，保持通道数为 64, 形状变为 [B, 64, W']

        # 3. 经过 1D 卷积层
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=64):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(3, stride=2, padding=1), nn.Dropout(0.3)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, stride=1, padding=1)
            layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


def make_first_layers(in_channels=1, out_channel=32):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.Conv2d(out_channel, out_channel, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.MaxPool2d((1, 3)),
        nn.Dropout(0.1),
        nn.Conv2d(out_channel, 64, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=(3, 6), stride=1, padding=(1, 1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d((2, 2)),  # 经过这一步，Height 维度通常会变小
        nn.Dropout(0.1)
    )


def get_model(num_classes, num_tabs, feature_dim, model_name="EM1", **kwargs):
    model = eval('RomerModel_{}'.format(model_name))(num_classes=num_classes, num_tabs=num_tabs,
                                                       feature_dim=feature_dim, **kwargs)
    return model


if __name__ == "__main__":
    from torchinfo import summary

    # 测试 RomerModel_NoOverlap
    B, C, T = 8, 20, 7200 # Batch size, Feature dim, Total length
    num_classes = 100
    
    print_colored("正在测试 RomerModel_NoOverlap...", "blue")
    model = get_model(num_classes=num_classes, num_tabs=5, feature_dim=C, 
                      model_name="NoOverlap", max_matrix_len=T, is_slice=True)
    
    # 测试输入结构 (B, 1, C, T)
    test_input = torch.randn(B, 1, C, T)
    output = model(test_input)
    
    print(f"\n输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape} (期望: [{B}, {num_classes}])\n")
    
    # 验证输出结构是否正确
    assert output.shape == (B, num_classes), f"输出形状错误！得到 {output.shape}"
    
    summary(model, input_size=(B, 1, C, T), col_names=["input_size", "output_size"], depth=3)
    print_colored("测试通过！", "green")

