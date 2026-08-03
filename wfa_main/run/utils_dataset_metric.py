# -*- coding: utf-8 -*-

# @Author: Xianjun Li
# @E-mail: xjli@mail.hnust.edu.cn
# @Date: 2025/12/1 下午8:22
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F
import numpy as np
from lxj_utils_sys import measurement, print_colored
from torch.utils.data import DataLoader


def autodict(**kwargs):
    return kwargs

def get_model(num_classes, config:dict, args:dict, **kwargs:dict):
    # 根据模型名称获取模型实例
    if config['model'] == 'Romer':
        from wfa_main.Model_Dataset import get_model
        model = get_model(num_classes=num_classes, feature_dim=kwargs['feat_dim'],
                          num_tabs=args['num_tabs'], model_name=args["Model_name"],
                          drop_path_rate=config['drop_path_rate'], depth=config['depth'],
                          max_matrix_len=config['max_matrix_len'],
                          overlap_ratio=config['overlap_ratio'],
                          embed_dim=config['embed_dim'],
                          )
    else:
        raise Exception(f"未知模型名称: {config['model']}")
    return model

def get_dataloader(X, y, config:dict, args:dict,num_workers=2, **kwargs:dict):
    # 根据模型名称获取数据集实例
    if config['model'] == "Romer":
        from wfa_main.Model_Dataset import TAMDataset as TrafficDataset
        print_colored(f">>>>>>>>>>> 注意，RoMER模型的TAM矩阵为: {args['TAM_type']} <<<<<<<<<<<<", "yellow")
        dataset_config = autodict(TAM_type=args['TAM_type'], traffic_length=config['seq_len'], load_ratio=args['load_ratio'],
                                  maximum_load_time=args['maximum_load_time'], use_idx=args['use_idx'],
                                  N_matrix=config['max_matrix_len'],
                                  bin_count=config['bin_count'],
                                  )
    else:
        raise Exception(f"未知模型名称: {config['model']}")

    dataset = TrafficDataset(X, y, **dataset_config)
    data_loader = DataLoader(dataset, batch_size=int(config['batch_size']),
                         shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)
    if args['use_idx'] and config['model'] == "Romer":
        feat_dim = next(iter(dataset))[0][0].shape[-2]
    else:
        feat_dim = next(iter(dataset))[0].shape[-2]

    return data_loader, feat_dim


def get_model_and_dataloader(X1,y1, X2,y2, num_classes, config:dict, args:dict):
    num_workers = args['num_workers']
    loader1, feat_dim = get_dataloader(X1, y1, config=config, args=args, num_workers=num_workers)
    loader2, _ = get_dataloader(X2, y2, config=config, args=args, num_workers=num_workers)
    model = get_model(config=config, args=args, num_classes=num_classes, feat_dim=feat_dim)
    print_colored(f"输入特征的维度: {feat_dim}", "yellow")
    return model, loader1, loader2

def load_data(data_path, max_load_time=None, remove_size=False):
    # 加载数据文件
    data = np.load(data_path)
    # 提取特征数据X和标签数据y
    X = data["X"]
    y = data["y"]
    # 时间负数调整：将时间数据取绝对值，确保所有时间值为正
    X[:, :, 0] = np.abs(X[:, :, 0])
    # 去除大小信息：此行被注释，原本可能用于保留数据的大小符号信息
    if remove_size:
        print_colored("数据集去除大小信息", "red")
        X[:, :, 1] = np.sign(X[:, :, 1])
    # 判断是否需要丢弃额外时间数据
    if max_load_time is not None:
        # 打印时间上限信息
        print(f"丢弃额外时间，时间上限：{max_load_time} s")
        # 找出超过时间上限的数据索引
        invalid_ind = X[:, :, 0]>max_load_time
        # 将超过时间上限的数据置为0
        X[invalid_ind, :] = 0
    else:
        # 如果不丢弃额外时间，打印加载完整流量的信息
        print("加载完整流量!")
    # 返回处理后的特征数据X和标签数据y
    return X, y

def knn_monitor(net, device, memory_data_loader, test_data_loader, num_classes, k=200, t=0.1):
    """
    Perform k-Nearest Neighbors (kNN) monitoring.

    Parameters:
    net (nn.Module): The neural network model.
    device (torch.device): The device to run the computations on.
    memory_data_loader (DataLoader): DataLoader for the memory bank.
    test_data_loader (DataLoader): DataLoader for the test data.
    num_classes (int): Number of classes.
    k (int): Number of nearest neighbors to use.
    t (float): Temperature parameter for scaling.

    Returns:
    tuple: True labels and predicted labels.
    """
    net.eval()
    total_num = 0
    feature_bank, feature_labels = [], []
    y_pred = []
    y_true = []

    with torch.no_grad():
        # Generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.to(device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            feature_labels.append(target)

        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous().to(device)
        feature_labels = torch.cat(feature_labels, dim=0).t().contiguous().to(device)

        # Loop through test data to predict the label by weighted kNN search
        for data, target in test_data_loader:
            data, target = data.to(device), target.to(device)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            pred_labels = knn_predict(feature, feature_bank, feature_labels, num_classes, k, t)
            total_num += data.size(0)
            y_pred.append(pred_labels[:, 0].cpu().numpy())
            y_true.append(target.cpu().numpy())

    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()

    return y_true, y_pred


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    """
    Predict labels using k-Nearest Neighbors (kNN) with cosine similarity.

    Parameters:
    feature (Tensor): Feature tensor.
    feature_bank (Tensor): Feature bank tensor.
    feature_labels (Tensor): Labels corresponding to the feature bank.
    classes (int): Number of classes.
    knn_k (int): Number of nearest neighbors to use.
    knn_t (float): Temperature parameter for scaling.

    Returns:
    Tensor: Predicted labels.
    """
    feature_labels = feature_labels.long()

    sim_matrix = torch.mm(feature, feature_bank)
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)

    return pred_labels

def compute_metric(y_true_fine, y_pred_fine):
    y_true_fine = y_true_fine.reshape(-1, y_true_fine.shape[-1])
    y_pred_fine = y_pred_fine.reshape(-1, y_pred_fine.shape[-1])

    num_classes = np.max(y_true_fine) + 1
    y_true_fine = gen_one_hot(y_true_fine, num_classes)
    y_pred_fine = gen_one_hot(y_pred_fine, num_classes)

    result = measurement(y_true_fine, y_pred_fine, eval_metrics="all")
    return result

def gen_one_hot(arr, num_classes):
    binary = np.zeros((arr.shape[0], num_classes))
    for i in range(arr.shape[0]):
        binary[i, arr[i]] = 1
    return binary

if __name__ == "__main__":
    from lxj_utils_sys import get_func_use_dic_fields
    print(get_func_use_dic_fields([get_dataloader], target_id="config"))
    print(get_func_use_dic_fields([get_model], target_id="config"))