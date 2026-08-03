import torch
from torch import nn
from timm.models.layers import trunc_normal_
from timm.models.layers import DropPath, Mlp
import torch.nn.functional as F

import torch
import torch.nn as nn


class Cross_MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_mhsa_layers, dim_feedforward):
        super().__init__()
        drop_path_rate = 0.1
        self.nets = nn.ModuleList(
            [Cross_MHSA_Block(embed_dim, num_heads, dim_feedforward) for _ in range(num_mhsa_layers)]
        )
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, q, k, v, pos_embed=0):
        """
        q: Query 序列 (例如：需要被注入信息的流量统计特征序列)
        k: Key 序列 (例如：包含上下文信息的字节流特征)
        v: Value 序列 (通常与 k 相同)
        pos_embed: 针对 Query 的位置编码
        """
        output = q + pos_embed

        # 逐层传递，每一层的 Query (output) 在不断更新，而 Key 和 Value (k, v) 保持不变
        for layer in self.nets:
            output = output + self.drop_path(layer(output, k, v))

        return output


class Cross_MHSA_Block(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward):
        super().__init__()
        drop_path_rate = 0.1
        self.drop_path = DropPath(drop_path_rate)

        # 1. 交叉注意力部分 (Cross-Attention)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=nhead, batch_first=True)

        # 2. 前馈神经网络部分 (MLP)
        self.norm2 = nn.LayerNorm(embed_dim)
        # 假设 Mlp 已经在外部定义
        self.mlp = Mlp(in_features=embed_dim, hidden_features=dim_feedforward, act_layer=nn.GELU, drop=0.1)

    def forward(self, q, k, v):
        # 步骤 1: Cross-Attention
        # 对 q 进行 LayerNorm 后作为 Query，外部传入的 k 和 v 作为 Key 和 Value
        norm_q = self.norm1(q)
        attn_output, _ = self.cross_attn(query=norm_q, key=k, value=v)

        # 残差连接
        q = q + self.drop_path(attn_output)

        # 步骤 2: MLP 层
        # 对当前的 q 进行 LayerNorm 后输入 MLP，再进行残差连接
        q = q + self.drop_path(self.mlp(self.norm2(q)))

        return q

class MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_mhsa_layers, dim_feedforward, atten_config):
        super().__init__()
        drop_path_rate = 0.1
        self.nets = nn.ModuleList(
            [MHSA_Block(embed_dim, num_heads, dim_feedforward, atten_config) for _ in range(num_mhsa_layers)])
        self.drop_path = DropPath(drop_path_rate)
    def forward(self, x, pos_embed=0):
        output = x + pos_embed

        for layer in self.nets:
            output = output + self.drop_path(layer(output))
        return output

class MHSA_Block(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, atten_config):
        super().__init__()
        drop_path_rate = 0.1
        self.attn = eval(f'Attention_{atten_config["name"]}')(dim=embed_dim, num_heads=nhead, **atten_config)
        # Attention_casual
        # Attention_TopM
        # Attention_Liear
        # Attention_base
        self.drop_path = DropPath(drop_path_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=dim_feedforward, act_layer=nn.GELU, drop=0.1)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention_TopM(nn.Module):
    def __init__(self, dim, num_heads, dropout, top_m=-1, **kwargs):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.top_m = top_m

        self.qkv = nn.Linear(dim, dim * 3)
        # self.attn_drop = nn.Sequential(
        #     nn.Softmax(dim=-1),
        #     nn.Dropout(dropout),
        # )
        # self.proj_drop = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.Dropout(dropout),
        # )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.top_m == -1:
            pass
        else:
            mask = torch.zeros(B, self.num_heads, N, N, device=q.device, requires_grad=False)
            index = torch.topk(attn, k=min(self.top_m,attn.shape[-1]), dim=-1, largest=True)[1]
            mask.scatter_(-1, index, 1.)
            attn = torch.where(mask > 0, attn, torch.full_like(attn, float('-inf')))

        #attn = self.attn_drop(attn)
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        #x = self.proj_drop(x)
        return x


class Attention_Causal(nn.Module):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)

        # 如果需要输出投影层，可以取消注释
        # self.proj = nn.Linear(dim, dim)
        # self.proj_drop = nn.Dropout(0.1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape

        # 1. 计算 Q, K, V
        # Shape: (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. 计算注意力分数 (Attention Scores)
        # Shape: (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # ==================== 核心修改：添加因果掩码 ====================
        # 生成一个下三角矩阵 (Lower Triangular Matrix)
        # torch.tril 会将上三角（不包含对角线）置为 0，下三角和对角线保持为 1
        # mask shape: (N, N)
        mask = torch.ones(N, N, device=x.device).tril()

        # 使用 masked_fill 将 mask 为 0 的位置（即未来位置）填充为 -inf
        # 这样 Softmax 之后，这些位置的概率会趋近于 0
        attn = attn.masked_fill(mask == 0, float('-inf'))
        # ==============================================================

        # 3. 归一化 (Softmax)
        attn = F.softmax(attn, dim=-1)

        # 4. 聚合 Value
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 如果有输出投影层
        # x = self.proj(x)
        # x = self.proj_drop(x)

        return x

class Attention_base(nn.Module):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        # self.attn_drop = nn.Sequential(
        #     nn.Softmax(dim=-1),
        # )
        # self.proj_drop = nn.Sequential(
        #     nn.Linear(dim, dim),
        # )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x

class Attention_Linear(nn.Module):
    def __init__(self,dim: int, num_heads: int, approx_dim: int, use_bias: bool = True, **kwargs):
        """
        多头线性注意力机制

        参数:
        embed_dim: 输入特征维度
        value_dim: 值向量维度（输出维度）
        approx_dim: 近似注意力维度
        num_heads: 注意力头数
        use_bias: 是否在特征映射中使用偏置项
        """
        super().__init__()
        embed_dim = dim
        value_dim = dim
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.approx_dim = approx_dim
        self.num_heads = num_heads
        self.use_bias = use_bias

        # 维度校验
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if value_dim % num_heads != 0:
            raise ValueError(f"value_dim ({value_dim}) must be divisible by num_heads ({num_heads})")

        # 计算每个头的维度
        self.head_dim = embed_dim // num_heads
        self.value_head_dim = value_dim // num_heads

        # QKV投影层
        self.qkv_proj = nn.Linear(embed_dim, (embed_dim * 2) + value_dim)

        # 特征映射矩阵 - 可选择是否使用偏置
        self.proj = nn.Linear(self.head_dim, approx_dim, bias=use_bias)
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化投影层权重
        nn.init.xavier_uniform_(self.proj.weight)
        # 如果使用偏置，初始化偏置项
        if self.use_bias:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        total_heads = batch_size * self.num_heads  # 计算总头数

        # 生成Q,K,V
        qkv = self.qkv_proj(x)
        Q, K, V = torch.split(qkv, [self.embed_dim, self.embed_dim, self.value_dim], dim=-1)

        # 分割多头并调整维度
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.value_head_dim).permute(0, 2, 1, 3)

        # 合并批次和头维度
        Q = Q.contiguous().view(total_heads, seq_len, self.head_dim)
        K = K.contiguous().view(total_heads, seq_len, self.head_dim)
        V = V.contiguous().view(total_heads, seq_len, self.value_head_dim)

        # 特征映射 φ(x) = √(1 + (Wx + b)^2) 或 √(1 + (Wx)^2)
        projected_Q = self.proj(Q)
        projected_K = self.proj(K)
        phi_Q = torch.sqrt(1 + projected_Q ** 2)
        phi_K = torch.sqrt(1 + projected_K ** 2)

        # 计算分子项: φ(Q)·(φ(K)^T·V)
        phi_K_t = phi_K.transpose(1, 2)  # [total_heads, approx_dim, seq_len]
        KTV = torch.bmm(phi_K_t, V)  # [total_heads, approx_dim, value_head_dim]
        numerator = torch.bmm(phi_Q, KTV)  # [total_heads, seq_len, value_head_dim]

        # 计算分母项: φ(Q)·(φ(K)^T·1)
        ones = torch.ones(total_heads, seq_len, 1, device=phi_K.device)  # [total_heads, seq_len, 1]
        KTO = torch.bmm(phi_K_t, ones)  # [total_heads, approx_dim, 1]
        denominator = torch.bmm(phi_Q, KTO)  # [total_heads, seq_len, 1]

        # 归一化输出
        epsilon = 1e-6
        head_output = numerator / (denominator + epsilon)

        # 合并多头输出
        head_output = head_output.view(batch_size, self.num_heads, seq_len, self.value_head_dim)
        output = head_output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, seq_len, self.value_dim)

        return output