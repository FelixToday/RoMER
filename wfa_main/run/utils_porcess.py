# -*- coding: utf-8 -*-
# @Author: Xianjun Li
# @E-mail: xjli@mail.hnust.edu.cn
# @Date: 2025/12/11 下午7:25

from tqdm import tqdm
import argparse
import os
import torch
import numpy as np

from wfa_main.run.const import dataset_lib, get_filebase_dir, get_machine_name
import configparser
from lxj_utils_sys import parse_args, str_to_bool, print_colored
from wfa_main.run.utils_dataset_metric import load_data, measurement, knn_monitor, compute_metric
from lxj_utils_sys import measurement_multilabel

param_description = {
    "model": "模型名称",
    "device": "训练设备",
    "seq_len": "输入序列的最大长度",
    "train_epochs": "训练的总轮数",
    "batch_size": "每个批次的样本数量",
    "learning_rate": "学习率",
    "optimizer": "优化器",
    "eval_metrics": "评估指标",
    "save_metric": "选择最佳模型的指标",
    "max_matrix_len": "特征矩阵的最大长度",
    "log_transform": "是否对特征进行对数变换",
    "embed_dim": "嵌入向量的维度",
    "time_interval_threshold": "判断簇的百分比",
    "maximum_cell_number": "数据包cell分级数量",
    "num_heads": "注意力头的数量",
    "r_of_lina": "Lina中的关键参数 `r`",
    "atten_type": "使用的注意力机制类型",
    "num_tabs": "标签页数量",
    "sample_num": "样本数量设置",
}


def get_parser(mode="train"):
    """
    根据模式创建参数解析器

    Args:
        mode (str): 模式选择，可选 "train" 或 "test"

    Returns:
        argparse.ArgumentParser: 配置好的参数解析器
    """
    parser = argparse.ArgumentParser(description=f"参数配置 - {mode}模式")

    # ========== 公共参数 ==========
    # 数据集相关
    parser.add_argument('--machine_name', type=str, default=get_machine_name(),
                        help='服务器名称')
    parser.add_argument('--config', default="config/Romer.ini",
                        help="模型配置文件路径")
    parser.add_argument('--checkpoint_path', default="../../checkpoints",
                        help="运行结果存储路径")
    parser.add_argument('--file_base_dir', default="auto",
                        help="数据集存储路径")
    parser.add_argument('--dataset', default="CW",
                        help="训练和评估使用的数据集名称")
    parser.add_argument('--load_ratio', type=float, default=100,
                        help="数据加载比例（百分比）")

    # 设备与运行配置
    parser.add_argument('--device', type=str, default='cuda:0',
                        help="训练设备，如：cuda:0, cpu")
    parser.add_argument('--note', type=str, default='test',
                        help="保存运行结果的文件夹")

    # 数据加载配置
    parser.add_argument('--maximum_load_time', type=float, default=80,
                        help="最大加载时间（秒）")
    parser.add_argument('--drop_extra_time', type=str_to_bool, default=True,
                        help="是否丢弃超出最大加载时间的数据")
    parser.add_argument('--num_workers', type=int, default=16,
                        help="数据加载的工作进程数")
    parser.add_argument('--remove_size', type=str_to_bool, default=False,
                        help="是否去除数据包的大小信息")

    # 模型相关
    parser.add_argument('--TAM_type', type=str, default='none',
                        help="提取特征的TAM方法")
    parser.add_argument('--Model_name', type=str, default='none',
                        help="模型名称")

    parser.add_argument('--use_idx', type=str, default='False',
                        help="是否使用idx作为模型输入")

    # parser.add_argument('--overlap_ratio', type=float, default=0,
    #                     help="重叠率")

    # 测试模式相关
    parser.add_argument('--is_Sen', type=str_to_bool, default=False,
                        help="是否为参数敏感性测试模式")
    parser.add_argument("--max_matrix_len", type=int, default=7200,
                        help=print_colored("最大矩阵长度", "blue", is_print=False))
    parser.add_argument("--overlap_ratio", type=float, default=0.6,
                        help=print_colored("重叠率", "green", is_print=False))
    parser.add_argument("--embed_dim", type=int, default=256,
                        help=print_colored("嵌入向量的维度", "green", is_print=False))
    parser.add_argument("--bin_count", type=int, default=5,
                        help=print_colored("bin数量", "green", is_print=False))

    parser.add_argument("--Sen_num_aug", type=int, default=30)
    parser.add_argument('--test_flag', type=str_to_bool, default=True,
                        help="是否打开测试模式")
    # ========== 模式特定参数 ==========
    if mode == "train":
        # 训练模式特有参数
        parser.add_argument('--train_epochs', type=int, default=30,
                            help="训练的总轮次")
        # 优化器相关
        parser.add_argument('--optim', type=str_to_bool, default=False,
                            help="是否使用优化配置参数")
        parser.add_argument('--weight_decay', type=float, default=0.05,
                            help="权重衰减系数，用于正则化")
        parser.add_argument('--min_lr', type=float, default=1e-6,
                            help="学习率的最小值")
        parser.add_argument('--warmup_epochs', type=int, default=5,
                            help="学习率预热轮数")
        parser.add_argument('--stag_epochs', type=int, default=20,
                            help="最大停滞次数")
        parser.add_argument('--valid_name', type=str, default='valid',
                            help="验证数据集名称")
    elif mode == "test":
        parser.add_argument('--is_pr_auc', type=str_to_bool, default=False,
                            help="是否使用PR-AUC作为评估指标")
        parser.add_argument('--first_check_ratio', type=str_to_bool, default=True,
                            help=print_colored("第1个检查的ratio", "green", is_print=False))
    else:
        raise ValueError(f"不支持的mode参数: {mode}，请使用 'train' 或 'test'")

    return parser

def model_forward(model, cur_X, config, args):
    """统一模型前向传播接口"""
    if config['model'] == 'Romer' and args['use_idx']:
        cur_X, idx = cur_X
        return model(cur_X, idx)
    else:
        return model(cur_X)

def move_to_device(cur_data, device, config, args):
    if config['model'] == 'Romer':
        cur_X, cur_y = cur_data[0][0].to(device), cur_data[1].to(device)
        if args['use_idx']:
            idx = cur_data[0][1].to(device)
        else:
            idx = None
        cur_X = cur_X, idx
    else:
        cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device)
    return cur_X, cur_y.long()

def adjust_system_args(args, config):
    """调整系统参数"""
    # 自动设置tab数量和最大加载时长
    args["num_tabs"] = dataset_lib[args['dataset']]['num_tabs']
    args["maximum_load_time"] = dataset_lib[args['dataset']]['maximum_load_time']

    if args['is_Sen']:
        config['max_matrix_len'] = args['max_matrix_len']
        config['embed_dim'] = args['embed_dim']
        config['overlap_ratio'] = args['overlap_ratio']
        config['bin_count'] = args['bin_count']



    # 自动设置文件基础目录
    if args['file_base_dir'] == "auto":
        args['file_base_dir'] = get_filebase_dir()

    # 如果是优化模式，需要调整一些参数
    if args.get('optim', False):
        args['train_epochs'] = 100
    # 如果是测试模式，就调整较小的epoch和数据量
    if args['test_flag']:
        args['train_epochs'] = 3
        args['sample_num'] = 800
    else:
        args['sample_num'] = -1

    # 调整序列长度和分窗口数（multi-tab 场景）
    config_name = str(os.path.basename(args['config'])).strip(".ini")
    if args['num_tabs'] > 1 and not args['is_Sen']:
        config['max_matrix_len'] = 7200
        config['seq_len'] = 10000


    return args, config

def load_config_file(config_path):
    """加载配置文件"""
    config = configparser.ConfigParser()
    config.read(config_path)
    return parse_args(config)[0]

def get_num_classes(test_y, args):
    """确定类别数量"""
    if args['num_tabs'] == 1:
        return len(np.unique(test_y))
    else:
        return test_y.shape[1]

def load_dataset_data(args, dataname, shuffle=True):
    """加载训练和验证数据"""
    train_path = os.path.join(args['file_base_dir'], args['dataset'], f"{dataname[0]}.npz")
    valid_path = os.path.join(args['file_base_dir'], args['dataset'], f"{dataname[1]}.npz")

    if args['drop_extra_time']:
        load_time = args['maximum_load_time']
    else:
        load_time = None
    X1, y1 = load_data(train_path, max_load_time=load_time, remove_size=args.get("remove_size", False))
    X2, y2 = load_data(valid_path, max_load_time=load_time, remove_size=args.get("remove_size", False))
    if args['sample_num'] > 0:
        if shuffle:
            idx = np.random.permutation(range(len(y1)))
            X1, y1 = X1[idx], y1[idx]
            idx = np.random.permutation(range(len(y2)))
            X2, y2 = X2[idx], y2[idx]
        X1, y1 = X1[:args['sample_num']], y1[:args['sample_num']]
        X2, y2 = X2[:args['sample_num']], y2[:args['sample_num']]

    print(f"{dataname[0]} 数据集样本数: {len(y1)}, {dataname[1]} 数据集样本数: {len(y2)}")


    return X1, y1, X2, y2

def evaluate_one_epoch(model, val_loader, device, config, args, num_classes):
    """执行单个验证轮次"""
    model.eval()

    if args['num_tabs'] > 1:
        # 多标签分类验证
        y_pred_score = np.zeros((0, num_classes))
        y_true = np.zeros((0, num_classes))

        with torch.no_grad():
            for cur_data in val_loader:
                cur_X, cur_y = move_to_device(cur_data, device, config, args)
                outs = model_forward(model, cur_X, config, args)
                y_pred_score = np.append(y_pred_score, outs.cpu().numpy(), axis=0)
                y_true = np.append(y_true, cur_y.cpu().numpy(), axis=0)

        # 计算多标签指标
        max_tab = 5
        tp = {tab: 0 for tab in range(1, max_tab + 1)}

        for idx in range(y_pred_score.shape[0]):
            cur_pred = y_pred_score[idx]
            for tab in range(1, max_tab + 1):
                target_webs = cur_pred.argsort()[-tab:]
                tp[tab] += sum(y_true[idx, target_web] > 0 for target_web in target_webs)

        mapk = 0.0
        valid_result = {}

        for tab in range(1, max_tab + 1):
            p_tab = tp[tab] / (y_true.shape[0] * tab)
            mapk += p_tab

            p_local = round(p_tab, 4) * 100
            ap_local = round(mapk / tab, 4) * 100
            print(f"p@{tab} {p_local:.4f}", f"ap@{tab} {ap_local:.4f}")
            valid_result.update({f'p@{tab}': p_local, f'ap@{tab}': ap_local})
            if tab == args['num_tabs']:
                p_global = round(p_tab, 4) * 100
                ap_global = round(mapk / tab, 4) * 100
                valid_result_final = {f'**P@{tab}': p_global, f'**MAP@{tab}': ap_global}

        # 粗粒度指标
        y_pred_coarse = y_pred_score.argsort()[:, -2:]
        y_true_coarse = [torch.nonzero(sample).squeeze().tolist() for sample in torch.tensor(y_true)]
        y_true_coarse = np.array(y_true_coarse)
        # coarse_result = compute_metric(y_true_coarse, y_pred_coarse)
        coarse_result = measurement_multilabel(y_true_coarse, y_pred_coarse, eval_metrics="Precision Recall F1-score")

        print(coarse_result)
        coarse_result = {"**" + key: value for key, value in coarse_result.items()}
        valid_result.update(coarse_result)
        valid_result.update(valid_result_final)
        return valid_result, ap_global

    else:
        # 单标签分类验证
        if config['model'] == "TF":
            valid_true, valid_pred = knn_monitor(model, device, val_loader, val_loader, num_classes, 10)
        else:
            valid_pred = []
            valid_true = []

            with torch.no_grad():
                for cur_data in tqdm(val_loader, dynamic_ncols=True):
                    cur_X, cur_y = move_to_device(cur_data, device, config, args)
                    outs = model_forward(model, cur_X, config, args)
                    cur_pred = torch.argsort(outs, dim=1, descending=True)[:, 0]

                    valid_pred.append(cur_pred.cpu().numpy())
                    valid_true.append(cur_y.cpu().numpy())

            valid_pred = np.concatenate(valid_pred)
            valid_true = np.concatenate(valid_true)

        valid_result = measurement(valid_true, valid_pred, eval_metrics="all")
        return valid_result, valid_result["F1-score"]