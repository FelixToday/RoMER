import os
import json
import math
import torch
import numpy as np
from typing import Union, List, Dict, Optional, Any
from .utils import print_colored

def calculate_conv_output_size(input_size, conv_settings):
    """
    计算卷积网络的输出尺寸

    Args:
        input_size: 输入尺寸
        conv_settings: 卷积设置，可以是：
            - 列表/元组: [kernel, stride, pad, dialate] (单层)
            - 列表的列表: [[k1,s1,p1,d1], [k2,s2,p2,d2], ...] (多层)
    """
    # 判断输入类型
    if isinstance(conv_settings[0], (list, tuple)):
        # 多层卷积：[[k1,s1,p1,d1], [k2,s2,p2,d2], ...]
        settings_list = conv_settings
    else:
        # 单层卷积：[kernel, stride, pad, dialate]
        settings_list = [conv_settings]

    current_size = input_size

    for setting in settings_list:
        if len(setting) != 4:
            raise ValueError("每个卷积设置必须包含4个参数: [kernel, stride, pad, dialate]")

        k, s, p, d = setting

        # 处理same填充
        if p == "same":
            p = k // 2

        # 确保所有参数都是数值类型
        if not all(isinstance(x, (int, float)) for x in [k, s, p, d]):
            raise TypeError("计算时所有参数必须是数值类型")

        # 计算当前层的输出尺寸
        current_size = (current_size + 2 * p - d * (k - 1) - 1) // s + 1

    return current_size




class LearningRateScheduler:
    def __init__(self, optimizer, lr, min_lr, warmup_epochs, total_epochs):
        """
        学习率调度器

        Args:
            optimizer: 优化器
            lr: 基础学习率
            min_lr: 最小学习率
            warmup_epochs: 预热轮数
            total_epochs: 总训练轮数
        """
        self.optimizer = optimizer
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def step(self, epoch):
        """
        根据当前epoch调整学习率

        Args:
            epoch: 当前训练轮数

        Returns:
            lr: 当前学习率
        """
        if epoch < self.warmup_epochs:
            # 预热阶段：线性增加学习率
            lr = self.lr * epoch / self.warmup_epochs
        else:
            # 余弦退火阶段
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * \
                 (1. + math.cos(math.pi * (epoch - self.warmup_epochs) /
                                (self.total_epochs - self.warmup_epochs)))

        # 更新优化器中的学习率
        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr

        return lr




# def compute_pr_result(model, dataloader, task_type="binary", average="macro", device=None, downsample=True, num_points=100):
#     """
#     通用PR计算函数，支持二分类、多分类、多标签任务，以及宏/微平均。
#     可选择下采样 PR 曲线点，减少保存数据量。
#     """
#     import torch
#     import numpy as np
#     from sklearn.metrics import precision_recall_curve, auc
#
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#     all_labels = []
#     all_probs = []
#
#     with torch.no_grad():
#         for x, y in dataloader:
#             x, y = x.to(device), y.to(device)
#             outputs = model(x)
#             if task_type == "binary":
#                 probs = torch.sigmoid(outputs).cpu()
#             elif task_type == "multiclass":
#                 probs = torch.softmax(outputs, dim=1).cpu()
#             elif task_type == "multilabel":
#                 probs = torch.sigmoid(outputs).cpu()
#             else:
#                 raise ValueError("Unsupported task_type")
#
#             all_probs.append(probs)
#             all_labels.append(y.cpu())
#
#     y_true = torch.cat(all_labels).numpy()
#     y_score = torch.cat(all_probs).numpy()
#
#     result = {}
#
#     def downsample_curve(precision, recall, thresholds=None):
#         """下采样函数"""
#         if not downsample:
#             return precision, recall, thresholds
#         # 使用插值统一长度
#         all_points = num_points
#         mean_recall = np.linspace(0, 1, all_points)
#         mean_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
#         if thresholds is not None:
#             # thresholds长度比 precision/recall少1，用线性插值近似
#             thresholds_full = np.concatenate(([0], thresholds))
#             mean_thresholds = np.interp(mean_recall, recall[::-1], thresholds_full[::-1])
#             return mean_precision, mean_recall, mean_thresholds
#         return mean_precision, mean_recall, thresholds
#
#     if task_type == "binary":
#         precision, recall, thresholds = precision_recall_curve(y_true, y_score)
#         precision, recall, thresholds = downsample_curve(precision, recall, thresholds)
#         result["precision"] = precision.tolist()
#         result["recall"] = recall.tolist()
#         result["thresholds"] = thresholds.tolist() if thresholds is not None else []
#         result["auc"] = float(auc(recall, precision))
#
#     elif task_type == "multiclass":
#         n_classes = y_score.shape[1]
#         y_onehot = np.eye(n_classes)[y_true]
#         if average == "micro":
#             precision, recall, thresholds = precision_recall_curve(y_onehot.ravel(), y_score.ravel())
#             precision, recall, thresholds = downsample_curve(precision, recall, thresholds)
#             result["precision"] = precision.tolist()
#             result["recall"] = recall.tolist()
#             result["thresholds"] = thresholds.tolist() if thresholds is not None else []
#             result["auc"] = float(auc(recall, precision))
#         elif average == "macro":
#             precisions, recalls, aucs = [], [], []
#             for i in range(n_classes):
#                 p, r, t = precision_recall_curve(y_onehot[:, i], y_score[:, i])
#                 precisions.append(p)
#                 recalls.append(r)
#                 aucs.append(auc(r, p))
#             # 插值平均
#             mean_recall = np.linspace(0, 1, num_points)
#             mean_precision = np.zeros(num_points)
#             for p, r in zip(precisions, recalls):
#                 mean_precision += np.interp(mean_recall, r[::-1], p[::-1])
#             mean_precision /= n_classes
#             result["precision"] = mean_precision.tolist()
#             result["recall"] = mean_recall.tolist()
#             result["thresholds"] = []  # macro模式下无统一threshold
#             result["auc"] = float(np.mean(aucs))
#         else:
#             raise ValueError("average must be 'micro' or 'macro'")
#
#     elif task_type == "multilabel":
#         n_labels = y_score.shape[1]
#         if average == "micro":
#             precision, recall, thresholds = precision_recall_curve(y_true.ravel(), y_score.ravel())
#             precision, recall, thresholds = downsample_curve(precision, recall, thresholds)
#             result["precision"] = precision.tolist()
#             result["recall"] = recall.tolist()
#             result["thresholds"] = thresholds.tolist() if thresholds is not None else []
#             result["auc"] = float(auc(recall, precision))
#         elif average == "macro":
#             precisions, recalls, aucs = [], [], []
#             for i in range(n_labels):
#                 p, r, t = precision_recall_curve(y_true[:, i], y_score[:, i])
#                 precisions.append(p)
#                 recalls.append(r)
#                 aucs.append(auc(r, p))
#             # 插值平均
#             mean_recall = np.linspace(0, 1, num_points)
#             mean_precision = np.zeros(num_points)
#             for p, r in zip(precisions, recalls):
#                 mean_precision += np.interp(mean_recall, r[::-1], p[::-1])
#             mean_precision /= n_labels
#             result["precision"] = mean_precision.tolist()
#             result["recall"] = mean_recall.tolist()
#             result["thresholds"] = []
#             result["auc"] = float(np.mean(aucs))
#         else:
#             raise ValueError("average must be 'micro' or 'macro'")
#
#     return result

def compute_pr_result(model, dataloader, task_type="binary", average="macro", device=None, downsample=True,
                      num_points=100, fcn_move_to_device=None, fcn_model_forward=None):
    """
    通用PR计算函数，支持二分类、多分类、多标签任务，以及宏/微平均。
    可选择下采样 PR 曲线点，减少保存数据量。

    新增参数:
    - move_to_device: 自定义函数，用于将数据放到设备上。默认接收 (x, y, device)，需返回处理后的 (x, y)。
    - model_forward: 自定义函数，用于模型的前向传播。默认接收 (model, x)，需返回 outputs。
    """
    import torch
    import numpy as np
    from sklearn.metrics import precision_recall_curve, auc

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in dataloader:
            # 1. 动态设备转移处理
            if fcn_move_to_device is not None:
                x, y = fcn_move_to_device(x, y, device)
            else:
                x, y = x.to(device), y.to(device)

            # 2. 动态前向传播处理
            if fcn_model_forward is not None:
                outputs = fcn_model_forward(model, x)
            else:
                outputs = model(x)

            if task_type == "binary":
                probs = torch.sigmoid(outputs).cpu()
            elif task_type == "multiclass":
                probs = torch.softmax(outputs, dim=1).cpu()
            elif task_type == "multilabel":
                probs = torch.sigmoid(outputs).cpu()
            else:
                raise ValueError("Unsupported task_type")

            all_probs.append(probs)
            all_labels.append(y.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_score = torch.cat(all_probs).numpy()

    result = {}

    def downsample_curve(precision, recall, thresholds=None):
        """下采样函数"""
        if not downsample:
            return precision, recall, thresholds
        # 使用插值统一长度
        all_points = num_points
        mean_recall = np.linspace(0, 1, all_points)
        mean_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        if thresholds is not None:
            # thresholds长度比 precision/recall少1，用线性插值近似
            thresholds_full = np.concatenate(([0], thresholds))
            mean_thresholds = np.interp(mean_recall, recall[::-1], thresholds_full[::-1])
            return mean_precision, mean_recall, mean_thresholds
        return mean_precision, mean_recall, thresholds

    if task_type == "binary":
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        precision, recall, thresholds = downsample_curve(precision, recall, thresholds)
        result["precision"] = precision.tolist()
        result["recall"] = recall.tolist()
        result["thresholds"] = thresholds.tolist() if thresholds is not None else []
        result["auc"] = float(auc(recall, precision))

    elif task_type == "multiclass":
        n_classes = y_score.shape[1]
        y_onehot = np.eye(n_classes)[y_true]
        if average == "micro":
            precision, recall, thresholds = precision_recall_curve(y_onehot.ravel(), y_score.ravel())
            precision, recall, thresholds = downsample_curve(precision, recall, thresholds)
            result["precision"] = precision.tolist()
            result["recall"] = recall.tolist()
            result["thresholds"] = thresholds.tolist() if thresholds is not None else []
            result["auc"] = float(auc(recall, precision))
        elif average == "macro":
            precisions, recalls, aucs = [], [], []
            for i in range(n_classes):
                p, r, t = precision_recall_curve(y_onehot[:, i], y_score[:, i])
                precisions.append(p)
                recalls.append(r)
                aucs.append(auc(r, p))
            # 插值平均
            mean_recall = np.linspace(0, 1, num_points)
            mean_precision = np.zeros(num_points)
            for p, r in zip(precisions, recalls):
                mean_precision += np.interp(mean_recall, r[::-1], p[::-1])
            mean_precision /= n_classes
            result["precision"] = mean_precision.tolist()
            result["recall"] = mean_recall.tolist()
            result["thresholds"] = []  # macro模式下无统一threshold
            result["auc"] = float(np.mean(aucs))
        else:
            raise ValueError("average must be 'micro' or 'macro'")

    elif task_type == "multilabel":
        n_labels = y_score.shape[1]
        if average == "micro":
            precision, recall, thresholds = precision_recall_curve(y_true.ravel(), y_score.ravel())
            precision, recall, thresholds = downsample_curve(precision, recall, thresholds)
            result["precision"] = precision.tolist()
            result["recall"] = recall.tolist()
            result["thresholds"] = thresholds.tolist() if thresholds is not None else []
            result["auc"] = float(auc(recall, precision))
        elif average == "macro":
            precisions, recalls, aucs = [], [], []
            for i in range(n_labels):
                p, r, t = precision_recall_curve(y_true[:, i], y_score[:, i])
                precisions.append(p)
                recalls.append(r)
                aucs.append(auc(r, p))
            # 插值平均
            mean_recall = np.linspace(0, 1, num_points)
            mean_precision = np.zeros(num_points)
            for p, r in zip(precisions, recalls):
                mean_precision += np.interp(mean_recall, r[::-1], p[::-1])
            mean_precision /= n_labels
            result["precision"] = mean_precision.tolist()
            result["recall"] = mean_recall.tolist()
            result["thresholds"] = []
            result["auc"] = float(np.mean(aucs))
        else:
            raise ValueError("average must be 'micro' or 'macro'")

    return result

class IncrementalMetricCalculator:
    def __init__(self):
        self.confusion_matrix = np.zeros((0, 0), dtype=np.int64)

    def _to_numpy(self, data):
        if hasattr(data, "detach"):
            return data.detach().cpu().numpy()
        return np.array(data)

    def _expand_matrix(self, max_label):
        current_size = self.confusion_matrix.shape[0]
        target_size = int(max_label) + 1

        if target_size > current_size:
            new_matrix = np.zeros((target_size, target_size), dtype=np.int64)
            new_matrix[:current_size, :current_size] = self.confusion_matrix
            self.confusion_matrix = new_matrix

    def update(self, y_true_raw, y_pred_raw):
        y_true = np.atleast_1d(self._to_numpy(y_true_raw))
        y_pred = np.atleast_1d(self._to_numpy(y_pred_raw))

        if y_true.ndim > 1: y_true = np.argmax(y_true, axis=1)
        if y_pred.ndim > 1: y_pred = np.argmax(y_pred, axis=1)

        max_label = max(np.max(y_true), np.max(y_pred))
        self._expand_matrix(max_label)

        np.add.at(self.confusion_matrix, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)

    def get(self, round_num=None, mode="macro"):
        """
        获取计算指标。
        :param round_num: 保留小数位数，默认 2
        :param mode: "macro" 表示正常宏平均结果，"min" 表示最差类别的最小结果
        """
        cm = self.confusion_matrix
        if cm.size == 0:
            return {"Accuracy": 0.0, "Precision": 0.0, "Recall": 0.0, "F1-score": 0.0}

        tp = np.diag(cm)
        fp = np.sum(cm, axis=0) - tp
        fn = np.sum(cm, axis=1) - tp

        total_samples = np.sum(cm)
        tn = total_samples - (tp + fp + fn)

        present_classes = (np.sum(cm, axis=1) + np.sum(cm, axis=0)) > 0
        epsilon = 1e-9

        # 逐类别计算
        acc_per_class = (tp + tn) / (total_samples + epsilon)
        precision_per_class = tp / (tp + fp + epsilon)
        recall_per_class = tp / (tp + fn + epsilon)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + epsilon)

        if round_num is None:
            round_num = 2

        # 根据 mode 决定输出什么结果，但 Key 保持不变
        if mode == "min":
            return {
                "Accuracy": round(float(np.min(acc_per_class[present_classes])) * 100, round_num),
                "Precision": round(float(np.min(precision_per_class[present_classes])) * 100, round_num),
                "Recall": round(float(np.min(recall_per_class[present_classes])) * 100, round_num),
                "F1-score": round(float(np.min(f1_per_class[present_classes])) * 100, round_num)
            }
        else:  # 默认的 "macro" 正常模式
            acc_global = np.sum(tp) / (total_samples + epsilon)
            return {
                "Accuracy": round(float(acc_global) * 100, round_num),
                "Precision": round(float(np.mean(precision_per_class[present_classes])) * 100, round_num),
                "Recall": round(float(np.mean(recall_per_class[present_classes])) * 100, round_num),
                "F1-score": round(float(np.mean(f1_per_class[present_classes])) * 100, round_num)
            }

    def reset(self):
        self.confusion_matrix = np.zeros((0, 0), dtype=np.int64)


def measurement(y_true, y_pred, eval_metrics="all", mode="macro", round_num=2):
    # 1. 鲁棒的数据转换
    def _to_numpy(data):
        if hasattr(data, "detach"): return data.detach().cpu().numpy()
        return np.array(data)

    y_true = np.atleast_1d(_to_numpy(y_true))
    y_pred = np.atleast_1d(_to_numpy(y_pred))

    # 处理 One-hot 编码
    if y_true.ndim > 1: y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1: y_pred = np.argmax(y_pred, axis=1)

    # 2. 确定类别范围并构建混淆矩阵 (使用 bincount 实现矩阵填充，避免显式循环)
    max_label = int(max(np.max(y_true), np.max(y_pred)))
    num_classes = max_label + 1

    # 核心：通过向量化索引计算混淆矩阵
    # 索引公式：y_true * 类别总数 + y_pred
    combined = y_true.astype(np.int64) * num_classes + y_pred.astype(np.int64)
    cm = np.bincount(combined, minlength=num_classes ** 2).reshape(num_classes, num_classes)

    # 3. 向量化计算 TP, FP, FN
    tp = np.diag(cm).astype(float)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp
    total_samples = np.sum(cm)

    # 识别当前存在的类别（忽略在真值和预测中都没出现的类别）
    present_mask = (np.sum(cm, axis=1) + np.sum(cm, axis=0)) > 0
    epsilon = 1e-9  # 防止除零

    # 4. 计算各指标数组 (Vectorized Operations)
    # 只有 Accuracy 是全局的，其余按类别计算
    precision_array = tp / (tp + fp + epsilon)
    recall_array = tp / (tp + fn + epsilon)
    f1_array = 2 * (precision_array * recall_array) / (precision_array + recall_array + epsilon)

    # 5. 格式化结果
    results = {}
    if eval_metrics == "all":
        metrics_list = ["Accuracy", "Precision", "Recall", "F1-score"]
    else:
        metrics_list = eval_metrics.split(" ")

    # 使用 numpy 的聚合函数替代循环逻辑
    agg_func = np.min if mode == "min" else np.mean

    if "Accuracy" in metrics_list:
        results["Accuracy"] = round(float(np.sum(tp) / (total_samples + epsilon)) * 100, round_num)

    if "Precision" in metrics_list:
        results["Precision"] = round(float(agg_func(precision_array[present_mask])) * 100, round_num)

    if "Recall" in metrics_list:
        results["Recall"] = round(float(agg_func(recall_array[present_mask])) * 100, round_num)

    if "F1-score" in metrics_list:
        results["F1-score"] = round(float(agg_func(f1_array[present_mask])) * 100, round_num)

    return results

def measurement_multilabel(y_true, y_pred, eval_metrics="all", mode="macro", round_num=2, threshold=0.5):
    # 1. 辅助函数：安全地获取数组或嵌套列表中的最大值
    def _get_max_val(data):
        try:
            return int(np.max(data))
        except:  # 兼容不定长嵌套列表 (list of lists)
            return int(max([max(row) if hasattr(row, '__iter__') else row for row in data]))

    # 2. 鲁棒的数据转换
    def _to_numpy(data):
        if hasattr(data, "detach"): data = data.detach().cpu().numpy()
        # 处理不定长数组的情况：转为 object array 或保留为 list
        try:
            return np.array(data, dtype=float)
        except ValueError:
            return data  # 保持 list of lists 状态

    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    # 3. 核心改进：自动探测输入格式
    # 如果数据中存在 > 1 的值，说明它是 "类别索引" (Class Indices) 而不是概率或 0/1 矩阵
    is_index_format = _get_max_val(y_true) > 1 or _get_max_val(y_pred) > 1

    if is_index_format:
        # --- 模式 A: 索引格式转换为 Multi-hot 矩阵 ---
        max_class = max(_get_max_val(y_true), _get_max_val(y_pred))
        num_classes = max_class + 1  # 类别总数

        def _to_multihot(data, n_classes):
            N = len(data)
            multihot = np.zeros((N, n_classes), dtype=int)
            for i, row in enumerate(data):
                if not hasattr(row, '__iter__'): row = [row]  # 兼容单元素
                # 过滤负数(通常用于padding)并填充矩阵
                valid_idx = [int(x) for x in row if int(x) >= 0]
                multihot[i, valid_idx] = 1
            return multihot

        y_true_bin = _to_multihot(y_true, num_classes)
        y_pred_bin = _to_multihot(y_pred, num_classes)
    else:
        # --- 模式 B: 标准的概率矩阵 / 0-1 矩阵 ---
        if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)

        y_pred_bin = (y_pred >= threshold).astype(int)
        y_true_bin = y_true.astype(int)

        if y_true_bin.shape != y_pred_bin.shape:
            raise ValueError(f"形状不匹配: 真值 {y_true_bin.shape} vs 预测 {y_pred_bin.shape}")

    # 4. 向量化计算每个标签的 TP, FP, FN, TN (按列聚合)
    tp = np.sum(y_true_bin * y_pred_bin, axis=0).astype(float)
    fp = np.sum((1 - y_true_bin) * y_pred_bin, axis=0).astype(float)
    fn = np.sum(y_true_bin * (1 - y_pred_bin), axis=0).astype(float)
    tn = np.sum((1 - y_true_bin) * (1 - y_pred_bin), axis=0).astype(float)

    epsilon = 1e-9

    # 5. 计算各指标
    precision_array = tp / (tp + fp + epsilon)
    recall_array = tp / (tp + fn + epsilon)
    f1_array = 2 * (precision_array * recall_array) / (precision_array + recall_array + epsilon)

    # 过滤掉真值和预测中都没出现过的类别（防止稀疏矩阵中大量 TN 拉高整体宏平均）
    present_mask = (np.sum(y_true_bin, axis=0) + np.sum(y_pred_bin, axis=0)) > 0
    if not np.any(present_mask): present_mask = np.ones_like(present_mask, dtype=bool)

    # 6. 格式化结果
    results = {}
    metrics_list = ["Accuracy", "Exact_Match", "Precision", "Recall",
                    "F1-score"] if eval_metrics == "all" else eval_metrics.split(" ")
    agg_func = np.min if mode == "min" else np.mean

    if "Accuracy" in metrics_list:
        label_accuracy = (tp + tn) / (tp + fp + fn + tn + epsilon)
        results["Accuracy"] = round(float(agg_func(label_accuracy[present_mask])) * 100, round_num)

    if "Exact_Match" in metrics_list:
        exact_match = np.mean(np.all(y_true_bin == y_pred_bin, axis=1))
        results["Exact_Match"] = round(float(exact_match) * 100, round_num)

    if "Precision" in metrics_list:
        results["Precision"] = round(float(agg_func(precision_array[present_mask])) * 100, round_num)

    if "Recall" in metrics_list:
        results["Recall"] = round(float(agg_func(recall_array[present_mask])) * 100, round_num)

    if "F1-score" in metrics_list:
        results["F1-score"] = round(float(agg_func(f1_array[present_mask])) * 100, round_num)

    return results


class ModelCheckpoint:
    def __init__(self, filename: str, mode: str = 'min', metric_name: str = 'metric', max_stagnation_epochs: Optional[int] = None):
        """
        模型检查点初始化
        :param filename: 保存文件路径（包含文件夹）
        :param mode: 评价指标类型，'min'表示越小越好，'max'表示越大越好
        :param metric_name: 评价指标名称，用于记录文件
        :param max_stagnation_epochs: 最大停滞epoch次数，用于早停检查
        """
        self.filename = filename
        self.mode = mode
        self.metric_name = metric_name
        self.max_stagnation_epochs = max_stagnation_epochs
        self.best_metric = float('inf') if mode == 'min' else -float('inf')
        self.metric_file = os.path.splitext(self.filename)[0] + '_metric.json'
        self.stagnation_count = 0  # 当前停滞epoch次数

        self.output_dict = {
                'best_metric': self.best_metric,
                'save_epoch': 0,
                'metric_name': self.metric_name,
                'Mode': self.mode,
                'Complete': False
            }

        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    def save(self, metric: float, model: torch.nn.Module, epoch: Optional[int] = None, final: bool = False) -> bool:
        """
        根据评价指标保存模型和metric记录，并检查停滞次数
        :param metric: 当前评价指标值
        :param model: 要保存的模型
        :param epoch: 当前轮次
        :return: 如果停滞次数达到最大值返回True，否则返回False
        """
        # 检查是否应该保存模型
        print(f"当前轮次: {epoch}")
        if (self.mode == 'min' and metric < self.best_metric) or \
                (self.mode == 'max' and metric > self.best_metric):
            self.best_metric = metric
            torch.save(model.state_dict(), self.filename)

            # 保存metric到json文件
            self.output_dict = {
                'best_metric': metric,
                'save_epoch': 0 if epoch is None else epoch,
                'metric_name': self.metric_name,
                'Mode': self.mode,
                'complete': final
            }
            self.write_to_json()

            print(f'模型已保存到: {self.filename}')
            print(f'指标记录已保存到: {self.metric_file}')
            print(f"保存了新模型（当前{self.metric_name}: {metric}）")
            # 重置停滞次数
            self.stagnation_count = 0
        else:
            print(f"未达到保存条件（当前{self.metric_name}: {metric}，最佳: {self.best_metric}）")
            self.output_dict['complete'] = final
            self.write_to_json()
            # 增加停滞次数
            self.stagnation_count += 1

        # 显示当前停滞次数
        print(f"停滞的epoch次数: {self.stagnation_count}/{self.max_stagnation_epochs if self.max_stagnation_epochs is not None else '无限制'}")

        # 检查是否达到最大停滞次数
        if self.max_stagnation_epochs is not None and self.stagnation_count >= self.max_stagnation_epochs:
            print(f"达到最大停滞epoch次数: {self.max_stagnation_epochs}")
            self.output_dict['complete'] = True
            self.write_to_json()
            return True
        else:
            return False

    def load(self, model: torch.nn.Module, device: str = "cpu"):
        """
        从文件中加载模型
        :param model: 要加载状态的模型
        :param device: 设备，如'cpu'或'cuda'
        :return: 加载的模型, 最佳指标值, 保存时的轮次
        """
        if os.path.exists(self.filename):
            model.to(device)
            model_state_dict = torch.load(self.filename, map_location=device, weights_only=True)
            model.load_state_dict(model_state_dict)

            with open(self.metric_file, 'r', encoding='utf-8') as f:
                output_dict = json.load(f)
                print_colored(f'模型已从: {self.filename} 加载，\n加载轮次 {output_dict.get("save_epoch", "N/A")}，{self.metric_name}: {output_dict.get("best_metric", "N/A"):.4f}', 'green')
                self.best_metric = output_dict.get("best_metric", self.best_metric)
                return model, self.best_metric, output_dict.get("save_epoch")
        else:
            print(f'模型文件: {self.filename} 不存在')
            return model.to(device), None, None

    def write_to_json(self):
        with open(self.metric_file, 'w', encoding='utf-8') as f:
            json.dump(self.output_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # 单层卷积 - 列表格式
    result1 = calculate_conv_output_size(32, [3, 1, 1, 1])
    print(result1)  # 输出: 32

    # 多层卷积 - 列表的列表格式
    result2 = calculate_conv_output_size(32, [
        [3, 1, "same", 1],  # 第一层
        [5, 2, 2, 1],  # 第二层
        [3, 1, 1, 1]  # 第三层
    ])
    print(result2)  # 输出: 15 (32→32→15→15)

    # 多层卷积 - 列表的列表格式
    stride = 1
    result2 = calculate_conv_output_size(1800, [
        [8, 2, 0, 1],  # 第一层
        [8, 2, 0, 1],  # 第一层
        [8, 2, 0, 1],  # 第一层
        [8, 2, 0, 1],  # 第一层
    ])
    print(result2)  # 输出: 15 (32→32→15→15)


    import torch # 模拟 PyTorch 输入
    from sklearn.metrics import f1_score

    y_true = torch.tensor([0, 1, 2, 2, 1])
    y_pred = torch.tensor([0, 2, 2, 2, 0])

    # 我们的方法
    my_res = measurement(y_true, y_pred, "F1-score", mode="macro")

    # Sklearn 方法
    sk_res = round(f1_score(y_true.numpy(), y_pred.numpy(), average="macro") * 100, 2)

    print(f"My F1: {my_res['F1-score']} | Sklearn F1: {sk_res}")
    # 输出：My F1: 44.44 | Sklearn F1: 44.44