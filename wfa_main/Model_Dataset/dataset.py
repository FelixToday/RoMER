from torch.utils.data import Dataset
import math
import numpy as np
from lxj_utils_sys import print_colored
def align_length(X, target_length):
    """
    向量化处理：对形状为 (N, L, C) 的张量进行对齐，并包含严格检查。
    """
    # 检查输入是否为 numpy 数组或类数组对象
    assert isinstance(X, (np.ndarray, list)), f"输入 X 必须是 numpy 数组，当前类型为 {type(X)}"
    if isinstance(X, list):
        X = np.array(X)

    # 检查维度：必须是 (N, L, C) 三维
    assert X.ndim == 3, f"输入 X 必须是 3 维数组 (N, L, C)，当前维度为 {X.ndim}"

    # 检查特征维度：假设你的特征数固定为 2
    assert X.shape[2] == 2, f"特征维度 (C) 预期为 2，当前为 {X.shape[2]}"

    # 检查目标长度：必须大于 0
    assert isinstance(target_length, int) and target_length > 0, "target_length 必须是正整数"

    curr_len = X.shape[1]

    if curr_len >= target_length:
        # 截断：[所有样本, 前 target_length 个时间步, 所有特征]
        return X[:, :target_length, :]
    else:
        # 补齐：只在第 2 维（index=1）的末尾补零
        pad_size = target_length - curr_len
        # pad_width 格式: ((维1前, 维1后), (维2前, 维2后), (维3前, 维3后))
        pad_width = ((0, 0), (0, pad_size), (0, 0))
        return np.pad(X, pad_width, mode="constant", constant_values=0.0)

class TAMDataset(Dataset):
    def __init__(self, X, Y, traffic_length, TAM_type, maximum_load_time=80, N_matrix=1800, load_ratio=100, **kwargs):
        # 执行向量化对齐，如果 X 维度不对，这里会直接抛出 AssertionError
        self.X = align_length(X, traffic_length)
        assert len(self.X) == len(Y), f"样本数({len(self.X)})与标签数({len(Y)})不匹配"
        self.Y = Y
        self.TAM_type = TAM_type
        # assert TAM_type in ["RF", "Mamba"]
        self.traffic_length = traffic_length
        self.maximum_load_time = maximum_load_time
        self.N_matrix = N_matrix
        self.load_ratio = load_ratio
        print_colored("数据集 -- 特征矩阵长度：{}".format(N_matrix), "yellow")
        # 优化：在初始化时预加载一次 JSON 阈值，避免在 process_data 遍历里高频重复 I/O 读写
        if TAM_type in ["ED3", "EDIAT"]:
            bin_count = kwargs.get('bin_count', 5)
            print_colored("数据集 -- bin 数量：{}".format(bin_count), "yellow")
            import json
            import os
            IAT_path = "../run/analyze/IAT.json"
            if os.path.exists(IAT_path):
                with open(IAT_path, 'r') as f:
                    d = json.load(f)
                # 根据选用的类型，选择不同的键名
                key_name = 'log' if TAM_type == "EDIAT" else 'entropy'
                bin_data = d[f'bin_{int(bin_count)}'][key_name]
                kwargs.setdefault('out_dt_thresholds', bin_data['out_dt_thresholds'])
                kwargs.setdefault('in_dt_thresholds', bin_data['in_dt_thresholds'])
            else:
                print_colored(" bin 文件不存在，请检查", "red")
                raise FileNotFoundError

        self.args = kwargs


    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        # TAM 的维度为 B*C*L
        x = self.X[index]
        y = self.Y[index]

        # 逻辑：如果输入了 load_ratio 就用输入的，否则用类自身的 self.load_ratio
        current_ratio = min(self.load_ratio, 100)
        T = x[:, 0]
        T_max = T.max()

        threshold = T_max * current_ratio / 100

        T = np.trim_zeros(T, "b")
        valid_index = np.where(T <= threshold)[0]
        x = x[valid_index, :]
        return self.process_data(x), y
    def process_data(self, data):
        t_seq = data[:, 0]
        l_seq = data[:, 1]

        T_max = t_seq.max()
        current_index = int(np.min([np.ceil(T_max / self.maximum_load_time * self.N_matrix), self.N_matrix]))

        # get_TAM_Mamba get_TAM_TF
        TAM = eval(f"TAM_{self.TAM_type}")(t_seq, l_seq, args={**self.args, "N_matrix": self.N_matrix, "maximum_load_time": self.maximum_load_time})

        if self.args.get("use_idx", False):
            return TAM.astype(np.float32), current_index
        else:
            return TAM.astype(np.float32)

def TAM_RF(t_seq, l_seq, args):
    N_matrix = args["N_matrix"]
    sequence = np.sign(l_seq) * t_seq
    maximum_load_time = args["maximum_load_time"]
    feature = np.zeros((2, N_matrix), dtype=np.float32)  # Initialize feature matrix
    for pack in sequence:
        if pack == 0:
            break  # End of sequence
        elif pack > 0:
            if pack >= maximum_load_time:
                feature[0, -1] += 1  # Assign to the last bin if it exceeds maximum load time
            else:
                idx = int(pack * (N_matrix - 1) / maximum_load_time)
                feature[0, idx] += 1
        else:
            pack = np.abs(pack)
            if pack >= maximum_load_time:
                feature[1, -1] += 1  # Assign to the last bin if it exceeds maximum load time
            else:
                idx = int(pack * (N_matrix - 1) / maximum_load_time)
                feature[1, idx] += 1
    return feature


def TAM_Mamba(t_seq, l_seq, args):
    # 统计窗口长度 返回datalen
    time_interval_threshold = args.get("time_interval_threshold", 0.1)
    maximum_cell_number = args.get("maximum_cell_number", 2)

    feature_dim = 2 * (maximum_cell_number + 2)
    feature = np.zeros((feature_dim, args["N_matrix"]))
    w = args["maximum_load_time"] / args["N_matrix"]
    time_interval = w * time_interval_threshold

    current_timestamps = []
    data_len = []
    current_index = 0
    for l_k, t_k in zip(l_seq, t_seq):
        if t_k == 0 and l_k == 0:
            break  # End of sequence
        d_k = int(np.sign(l_k))
        c_k = min(int(np.abs(l_k) // 512), maximum_cell_number)  # [0, C]

        fragment = 0 if d_k < 0 else 1
        i = 2 * c_k + fragment  # [0, 2C + 1]
        j = min(math.floor(t_k / w), args["N_matrix"] - 1)
        j = max(j, 0)
        feature[i, j] += 1

        if j != current_index:
            feature[2 * maximum_cell_number + 2, j] = max(j - current_index, 0)
            data_len.append(len(current_timestamps))
            delta_t = np.diff(current_timestamps)
            cluster_count = np.sum(delta_t > time_interval) + 1
            feature[2 * maximum_cell_number + 3, current_index] = cluster_count
            current_index = j
            current_timestamps = [t_k]
        else:
            current_timestamps.append(t_k)
    delta_t = np.diff(current_timestamps)
    data_len.append(len(current_timestamps))
    cluster_count = np.sum(delta_t > time_interval) + 1
    feature[2 * maximum_cell_number + 3, current_index] = cluster_count
    return feature

def TAM_ED1(t_seq, l_seq, args):
    """
    优化版本特征提取函数，保留五个特征：
    0: 上行包数量
    1: 下行包数量
    2: 包间隔（窗口索引差）
    3: 上行包大小和（除以512）
    4: 下行包大小和（除以512）
    """
    # 初始化特征矩阵 (5行 x max_column列)
    maximum_load_time, N_matrix = args["maximum_load_time"], args["N_matrix"]
    feature = np.zeros((5, N_matrix))

    # 截断T==0之后的数据
    indices = np.flatnonzero(t_seq)
    ind = indices[-1]+1 if indices.size > 0 else len(t_seq)
    t_seq = t_seq[:ind]
    l_seq = l_seq[:ind]

    # 无有效数据时返回空特征
    if len(t_seq) == 0:
        return feature

    # 计算所有窗口索引
    all_windows = np.floor(t_seq / maximum_load_time * (N_matrix - 1)).astype(int)
    all_windows = np.clip(all_windows, 0, N_matrix - 1)

    # 特征0&1: 分别计算上行/下行包数量
    up_mask = l_seq > 0
    down_mask = l_seq < 0
    if np.any(up_mask):
        feature[0] = np.bincount(all_windows[up_mask], minlength=N_matrix)
    if np.any(down_mask):
        feature[1] = np.bincount(all_windows[down_mask], minlength=N_matrix)

    # 特征3&4: 分别计算上行/下行包大小和（除以512）
    if np.any(up_mask):
        # 上行包大小和：直接使用L[up_mask]（正数），然后除以512
        up_size_sum = np.bincount(all_windows[up_mask], weights=l_seq[up_mask], minlength=N_matrix)
        feature[3] = up_size_sum / 512.0  # 使用浮点除法避免整数截断
    if np.any(down_mask):
        # 下行包大小和：取-L[down_mask]（绝对值），然后除以512
        down_size_sum = np.bincount(all_windows[down_mask], weights=-l_seq[down_mask], minlength=N_matrix)
        feature[4] = down_size_sum / 512.0

    # 特征2: 包间隔（窗口索引差）
    unique_windows = np.unique(all_windows)
    if unique_windows.size > 1:
        window_gaps = np.diff(unique_windows)
        feature[2, unique_windows[1:]] = window_gaps

    return np.log(1+feature)

def TAM_ED2(t_seq, l_seq, args):
    """
    优化版本特征提取函数，保留五个特征：
    0: 上行包数量
    1: 下行包数量
    2: 上行包大小和（除以512）
    3: 下行包大小和（除以512）
    """
    # 初始化特征矩阵 (5行 x max_column列)
    maximum_load_time, N_matrix = args["maximum_load_time"], args["N_matrix"]
    feature = np.zeros((5, N_matrix))

    # 截断T==0之后的数据
    indices = np.flatnonzero(t_seq)
    ind = indices[-1]+1 if indices.size > 0 else len(t_seq)
    t_seq = t_seq[:ind]
    l_seq = l_seq[:ind]

    # 无有效数据时返回空特征
    if len(t_seq) == 0:
        return feature

    # 计算所有窗口索引
    all_windows = np.floor(t_seq / maximum_load_time * (N_matrix - 1)).astype(int)
    all_windows = np.clip(all_windows, 0, N_matrix - 1)

    # 特征0&1: 分别计算上行/下行包数量
    up_mask = l_seq > 0
    down_mask = l_seq < 0
    if np.any(up_mask):
        feature[0] = np.bincount(all_windows[up_mask], minlength=N_matrix)
    if np.any(down_mask):
        feature[1] = np.bincount(all_windows[down_mask], minlength=N_matrix)

    # 特征3&4: 分别计算上行/下行包大小和（除以512）
    if np.any(up_mask):
        # 上行包大小和：直接使用L[up_mask]（正数），然后除以512
        up_size_sum = np.bincount(all_windows[up_mask], weights=l_seq[up_mask], minlength=N_matrix)
        feature[3] = up_size_sum / 512.0  # 使用浮点除法避免整数截断
    if np.any(down_mask):
        # 下行包大小和：取-L[down_mask]（绝对值），然后除以512
        down_size_sum = np.bincount(all_windows[down_mask], weights=-l_seq[down_mask], minlength=N_matrix)
        feature[4] = down_size_sum / 512.0
    return np.log(1+feature)

def TAM_ED3(t_seq, l_seq, args):
    """
    向量化优化的 TAM_ED3，利用 Numpy 原生操作消灭 For 循环。
    窗口内特征：
        包含时间特征：
            上下行的dt分布
        包含空间特征：
            上下行的bs平均大小和数量
            上下行的数据包数量
    窗口间关联特征：
        当前窗口距离前多少个窗口为空白
    """
    # 1. 动态参数解析
    MAX_TIME = args["maximum_load_time"]
    NUM_WINDOWS = args['N_matrix']
    WINDOW_SIZE = MAX_TIME / NUM_WINDOWS

    t_thresh_out = args['out_dt_thresholds']# np.asarray(args.get('out_dt_thresholds', [0.0000, 8.8889, 17.7778, 26.6667, 35.5556, 44.4444]))
    t_thresh_in = args['in_dt_thresholds']#np.asarray(args.get('in_dt_thresholds', [0.0000, 8.8889, 17.7778, 26.6667, 35.5556, 44.4444]))

    n_t_out = len(t_thresh_out) - 1
    n_t_in = len(t_thresh_in) - 1

    total_dim = 2 + 4 + n_t_out + n_t_in + 1
    features = np.zeros((total_dim, NUM_WINDOWS))

    PKT_C_OUT = 0
    PKT_C_IN = 1
    BC_OUT = 2
    BS_AVG_OUT = 3
    BC_IN = 4
    BS_AVG_IN = 5
    TD_OUT_START = 6
    TD_IN_START = 6 + n_t_out
    GAP_IDX = total_dim - 1

    # ==========================
    # 数据预过滤与全局窗口映射
    # ==========================
    mask = (l_seq != 0) & (t_seq <= MAX_TIME)
    t_valid = t_seq[mask]
    l_valid = l_seq[mask]

    if len(t_valid) == 0:
        return features

    # 计算所有数据所属的窗口索引，防止因精度问题越界
    win_indices = np.floor(t_valid / WINDOW_SIZE).astype(int)
    win_indices = np.clip(win_indices, 0, NUM_WINDOWS - 1)

    # 提取上下行 Mask
    up_mask = (l_valid > 0)
    down_mask = (l_valid < 0)

    # ==========================
    # 特征 1: Packet Counts
    # ==========================
    if np.any(up_mask):
        features[PKT_C_OUT, :] = np.bincount(win_indices[up_mask], minlength=NUM_WINDOWS)[:NUM_WINDOWS]
    if np.any(down_mask):
        features[PKT_C_IN, :] = np.bincount(win_indices[down_mask], minlength=NUM_WINDOWS)[:NUM_WINDOWS]

    # ==========================
    # 特征 2: Window Gap
    # ==========================
    active_wins = np.unique(win_indices)
    if len(active_wins) > 0:
        gaps = np.empty_like(active_wins)
        gaps[0] = active_wins[0] - (-1) - 1  # 初始窗口与 -1 的距离
        if len(active_wins) > 1:
            gaps[1:] = np.diff(active_wins) - 1
        features[GAP_IDX, active_wins] = gaps

    # ==========================
    # 特征 3: Burst 特征计算 (平均值与数量)
    # Burst 核心定义: 方向改变 OR 窗口改变 时 Burst 打断
    # ==========================
    changes = np.where((l_valid[:-1] != l_valid[1:]) | (win_indices[:-1] != win_indices[1:]))[0] + 1
    splits = np.concatenate(([0], changes, [len(l_valid)]))

    burst_sizes = np.diff(splits)
    burst_wins = win_indices[splits[:-1]]
    burst_dirs = l_valid[splits[:-1]]

    for direction, bc_idx, bs_avg_idx in [(1, BC_OUT, BS_AVG_OUT), (-1, BC_IN, BS_AVG_IN)]:
        d_mask = (burst_dirs > 0) if direction == 1 else (burst_dirs < 0)
        if not np.any(d_mask):
            continue

        b_wins = burst_wins[d_mask]
        b_sizes = burst_sizes[d_mask]

        b_counts = np.bincount(b_wins, minlength=NUM_WINDOWS)[:NUM_WINDOWS]
        features[bc_idx, :] = b_counts

        b_size_sums = np.bincount(b_wins, weights=b_sizes, minlength=NUM_WINDOWS)[:NUM_WINDOWS]
        valid_wins = b_counts > 0
        features[bs_avg_idx, valid_wins] = b_size_sums[valid_wins] / b_counts[valid_wins]

    # ==========================
    # 特征 4: Time Diff 计算
    # ==========================
    for direction, td_start, t_thresh, n_t in [(1, TD_OUT_START, t_thresh_out, n_t_out), (-1, TD_IN_START, t_thresh_in, n_t_in)]:
        d_mask = up_mask if direction == 1 else down_mask
        if not np.any(d_mask):
            continue

        t_dir = t_valid[d_mask]
        w_dir = win_indices[d_mask]

        # 计算同向包内相邻包的 window 是否一致（跨窗口的不计算 diff）
        valid_diff = w_dir[:-1] == w_dir[1:]
        if not np.any(valid_diff):
            continue

        diffs_ms = (t_dir[1:] - t_dir[:-1])[valid_diff] * 1000.0
        diff_wins = w_dir[:-1][valid_diff]

        # 映射 Time Diffs 并批量更新矩阵
        t_indices = np.searchsorted(t_thresh, diffs_ms)
        t_indices = np.clip(t_indices, 0, n_t - 1)
        np.add.at(features, (td_start + t_indices, diff_wins), 1)

    return features


def TAM_Orig(t_seq, l_seq, args):
    """
    不提取特征，直接返回 1*7200 的特征矩阵。
    内容为 t 与 l 的符号相乘的结果 (sign(t) * sign(l))。
    多了截断到 7200，少了填充 0。
    """
    # 计算符号乘积
    # np.sign 对于正数返回 1，负数返回 -1，0 返回 0
    feature = np.sign(t_seq) * np.sign(l_seq)
    
    # 目标长度为 7200
    target_length = 7200
    
    # 截断或填充
    if len(feature) > target_length:
        feature = feature[:target_length]
    else:
        # 在末尾填充 0
        feature = np.pad(feature, (0, target_length - len(feature)), 'constant', constant_values=0)
    
    # 转换为 1*7200 的矩阵形式返回
    return feature.reshape(1, target_length)


# def TAM_ED3_abla(t_seq, l_seq, args):
#     """
#     向量化优化的 TAM_ED3，利用 Numpy 原生操作消灭 For 循环。
#     通过 args['input_feat'] 参数控制保留哪些特征：'dt', 'burst', 'count', 'win'
#     """
#     # 1. 动态参数解析
#     MAX_TIME = args["maximum_load_time"]
#     NUM_WINDOWS = args['N_matrix']
#     WINDOW_SIZE = MAX_TIME / NUM_WINDOWS
#
#     input_feat = args.get('input_feat', 'dt burst win count')
#     #print_colored("input_feat: %s" % input_feat, "blue")
#     t_thresh_out = np.asarray(args.get('out_dt_thresholds', [0.0000, 8.8889, 17.7778, 26.6667, 35.5556, 44.4444]))
#     t_thresh_in = np.asarray(args.get('in_dt_thresholds', [0.0000, 8.8889, 17.7778, 26.6667, 35.5556, 44.4444]))
#
#     n_t_out = len(t_thresh_out) - 1
#     n_t_in = len(t_thresh_in) - 1
#
#     total_dim = 2 + 4 + n_t_out + n_t_in + 1
#     features = np.zeros((total_dim, NUM_WINDOWS))
#
#     PKT_C_OUT = 0
#     PKT_C_IN = 1
#     BC_OUT = 2
#     BS_AVG_OUT = 3
#     BC_IN = 4
#     BS_AVG_IN = 5
#     TD_OUT_START = 6
#     TD_IN_START = 6 + n_t_out
#     GAP_IDX = total_dim - 1
#
#     selected_indices = []
#     if 'count' in input_feat:
#         selected_indices.extend([PKT_C_OUT, PKT_C_IN])
#     if 'burst' in input_feat or 'busrt' in input_feat:
#         selected_indices.extend([BC_OUT, BS_AVG_OUT, BC_IN, BS_AVG_IN])
#     if 'dt' in input_feat:
#         selected_indices.extend(list(range(TD_OUT_START, TD_IN_START + n_t_in)))
#     if 'win' in input_feat:
#         selected_indices.extend([GAP_IDX])
#
#     # ==========================
#     # 数据预过滤与全局窗口映射
#     # ==========================
#     mask = (l_seq != 0) & (t_seq <= MAX_TIME)
#     t_valid = t_seq[mask]
#     l_valid = l_seq[mask]
#
#     if len(t_valid) == 0:
#         return features[selected_indices, :]
#
#     # 计算所有数据所属的窗口索引，防止因精度问题越界
#     win_indices = np.floor(t_valid / WINDOW_SIZE).astype(int)
#     win_indices = np.clip(win_indices, 0, NUM_WINDOWS - 1)
#
#     # 提取上下行 Mask
#     up_mask = (l_valid > 0)
#     down_mask = (l_valid < 0)
#
#     # ==========================
#     # 特征 1: Packet Counts
#     # ==========================
#     if 'count' in input_feat:
#         if np.any(up_mask):
#             features[PKT_C_OUT, :] = np.bincount(win_indices[up_mask], minlength=NUM_WINDOWS)[:NUM_WINDOWS]
#         if np.any(down_mask):
#             features[PKT_C_IN, :] = np.bincount(win_indices[down_mask], minlength=NUM_WINDOWS)[:NUM_WINDOWS]
#
#     # ==========================
#     # 特征 2: Window Gap
#     # ==========================
#     if 'win' in input_feat:
#         active_wins = np.unique(win_indices)
#         if len(active_wins) > 0:
#             gaps = np.empty_like(active_wins)
#             gaps[0] = active_wins[0] - (-1) - 1  # 初始窗口与 -1 的距离
#             if len(active_wins) > 1:
#                 gaps[1:] = np.diff(active_wins) - 1
#             features[GAP_IDX, active_wins] = gaps
#
#     # ==========================
#     # 特征 3: Burst 特征计算 (平均值与数量)
#     # Burst 核心定义: 方向改变 OR 窗口改变 时 Burst 打断
#     # ==========================
#     if 'burst' in input_feat or 'busrt' in input_feat:
#         changes = np.where((l_valid[:-1] != l_valid[1:]) | (win_indices[:-1] != win_indices[1:]))[0] + 1
#         splits = np.concatenate(([0], changes, [len(l_valid)]))
#
#         burst_sizes = np.diff(splits)
#         burst_wins = win_indices[splits[:-1]]
#         burst_dirs = l_valid[splits[:-1]]
#
#         for direction, bc_idx, bs_avg_idx in [(1, BC_OUT, BS_AVG_OUT), (-1, BC_IN, BS_AVG_IN)]:
#             d_mask = (burst_dirs > 0) if direction == 1 else (burst_dirs < 0)
#             if not np.any(d_mask):
#                 continue
#
#             b_wins = burst_wins[d_mask]
#             b_sizes = burst_sizes[d_mask]
#
#             b_counts = np.bincount(b_wins, minlength=NUM_WINDOWS)[:NUM_WINDOWS]
#             features[bc_idx, :] = b_counts
#
#             b_size_sums = np.bincount(b_wins, weights=b_sizes, minlength=NUM_WINDOWS)[:NUM_WINDOWS]
#             valid_wins = b_counts > 0
#             features[bs_avg_idx, valid_wins] = b_size_sums[valid_wins] / b_counts[valid_wins]
#
#     # ==========================
#     # 特征 4: Time Diff 计算
#     # ==========================
#     if 'dt' in input_feat:
#         for direction, td_start, t_thresh, n_t in [(1, TD_OUT_START, t_thresh_out, n_t_out), (-1, TD_IN_START, t_thresh_in, n_t_in)]:
#             d_mask = up_mask if direction == 1 else down_mask
#             if not np.any(d_mask):
#                 continue
#
#             t_dir = t_valid[d_mask]
#             w_dir = win_indices[d_mask]
#
#             # 计算同向包内相邻包的 window 是否一致（跨窗口的不计算 diff）
#             valid_diff = w_dir[:-1] == w_dir[1:]
#             if not np.any(valid_diff):
#                 continue
#
#             diffs_ms = (t_dir[1:] - t_dir[:-1])[valid_diff] * 1000.0
#             diff_wins = w_dir[:-1][valid_diff]
#
#             # 映射 Time Diffs 并批量更新矩阵
#             t_indices = np.searchsorted(t_thresh, diffs_ms)
#             t_indices = np.clip(t_indices, 0, n_t - 1)
#             np.add.at(features, (td_start + t_indices, diff_wins), 1)
#
#     return features[selected_indices, :]
#
# def TAM_ED3_no_dt(t_seq, l_seq, args):
#     return TAM_ED3_abla(t_seq, l_seq, {**args, "input_feat": "burst win count"})
#
# def TAM_ED3_no_burst(t_seq, l_seq, args):
#     return TAM_ED3_abla(t_seq, l_seq, {**args, "input_feat": "dt win count"})
#
# def TAM_ED3_no_win(t_seq, l_seq, args):
#     return TAM_ED3_abla(t_seq, l_seq, {**args, "input_feat": "dt burst count"})
#
# def TAM_ED3_no_all(t_seq, l_seq, args):
#     return TAM_ED3_abla(t_seq, l_seq, {**args, "input_feat": "count"})
#

def TAM_EDRF(t_seq, l_seq, args):
    """
    向量化优化的 TAM_RF，按照 TAM_ED1 的形式提取特征
    """
    maximum_load_time, N_matrix = args["maximum_load_time"], args["N_matrix"]
    feature = np.zeros((2, N_matrix), dtype=np.float32)

    # 截断T==0之后的数据
    indices = np.flatnonzero(t_seq)
    ind = indices[-1]+1 if indices.size > 0 else len(t_seq)
    t_seq = t_seq[:ind]
    l_seq = l_seq[:ind]

    if len(t_seq) == 0:
        return feature

    all_windows = np.floor(t_seq / maximum_load_time * (N_matrix - 1)).astype(int)
    all_windows = np.clip(all_windows, 0, N_matrix - 1)

    up_mask = l_seq > 0
    down_mask = l_seq < 0
    if np.any(up_mask):
        feature[0] = np.bincount(all_windows[up_mask], minlength=N_matrix)
    if np.any(down_mask):
        feature[1] = np.bincount(all_windows[down_mask], minlength=N_matrix)

    return feature

def TAM_EDIAT(t_seq, l_seq, args):
    """
    向量化优化的 TAM_EDIAT，只提取 TAM_ED3 中的特征 4 (Time Diff)
    """
    MAX_TIME = args["maximum_load_time"]
    NUM_WINDOWS = args['N_matrix']
    WINDOW_SIZE = MAX_TIME / NUM_WINDOWS

    t_thresh_out = np.asarray(args.get('out_dt_thresholds', [0.0000, 8.8889, 17.7778, 26.6667, 35.5556, 44.4444]))
    t_thresh_in = np.asarray(args.get('in_dt_thresholds', [0.0000, 8.8889, 17.7778, 26.6667, 35.5556, 44.4444]))

    n_t_out = len(t_thresh_out) - 1
    n_t_in = len(t_thresh_in) - 1

    total_dim = n_t_out + n_t_in
    features = np.zeros((total_dim, NUM_WINDOWS))

    TD_OUT_START = 0
    TD_IN_START = n_t_out

    # ==========================
    # 数据预过滤与全局窗口映射
    # ==========================
    mask = (l_seq != 0) & (t_seq <= MAX_TIME)
    t_valid = t_seq[mask]
    l_valid = l_seq[mask]

    if len(t_valid) == 0:
        return features

    # 计算所有数据所属的窗口索引，防止因精度问题越界
    win_indices = np.floor(t_valid / WINDOW_SIZE).astype(int)
    win_indices = np.clip(win_indices, 0, NUM_WINDOWS - 1)

    # 提取上下行 Mask
    up_mask = (l_valid > 0)
    down_mask = (l_valid < 0)

    # ==========================
    # 特征: Time Diff 计算
    # ==========================
    for direction, td_start, t_thresh, n_t in [(1, TD_OUT_START, t_thresh_out, n_t_out), (-1, TD_IN_START, t_thresh_in, n_t_in)]:
        d_mask = up_mask if direction == 1 else down_mask
        if not np.any(d_mask):
            continue

        t_dir = t_valid[d_mask]
        w_dir = win_indices[d_mask]

        # 计算同向包内相邻包的 window 是否一致（跨窗口的不计算 diff）
        valid_diff = w_dir[:-1] == w_dir[1:]
        if not np.any(valid_diff):
            continue

        diffs_ms = (t_dir[1:] - t_dir[:-1])[valid_diff] * 1000.0
        diff_wins = w_dir[:-1][valid_diff]

        # 映射 Time Diffs 并批量更新矩阵
        t_indices = np.searchsorted(t_thresh, diffs_ms)
        t_indices = np.clip(t_indices, 0, n_t - 1)
        np.add.at(features, (td_start + t_indices, diff_wins), 1)

    return features

if __name__ == '__main__':
    from wfa_main.run.const import get_filebase_dir
    from wfa_main.run.utils_dataset_metric import load_data
    import os, time
    from tqdm import tqdm
    from lxj_utils_sys import get_func_use_dic_fields, print_colored

    print(get_func_use_dic_fields(TAM_Mamba, "args", ["maximum_load_time", "N_matrix"]))
    test_num = 5
    if test_num == 1:
        # 检查多个提取方法的输出维度
        dataset = "Closed_2tab"
        base_dir = get_filebase_dir()  # Closed_5tab regulator_Closed_2tab
        data_path = os.path.join(base_dir, dataset, "test.npz")
        X, y = load_data(data_path,max_load_time=None)
        test_set = TAMDataset(X, y, 5000, "RF", 80, 1800, 100)
        print("RF: ", test_set[0][0].shape, test_set[0][1])
        test_set = TAMDataset(X, y, 5000, "ED1", 80, 1800, 100)
        print("ED1: ", test_set[0][0].shape, test_set[0][1])
        test_set = TAMDataset(X, y, 5000, "Mamba", 80, 1800, 100,
                              **{"maximum_cell_number":2,
                                 "time_interval_threshold": 0.1})
        print("Mamba: ", test_set[0][0].shape, test_set[0][1])
    elif test_num == 2:
        # 检查多个提取方法的提取效率
        dataset = "Closed_2tab"
        base_dir = get_filebase_dir()  # Closed_5tab regulator_Closed_2tab
        data_path = os.path.join(base_dir, dataset, "test.npz")
        X, y = load_data(data_path, max_load_time=80)
        test_set = TAMDataset(X, y, 5000, "RF", 80, 1800, 100)
        for tam_type in ['RF', 'Mamba', "ED1"]:
            print(f"TAM_type is {tam_type}")
            test_set = TAMDataset(X, y, 5000, tam_type, 80, 1800, 100,
                                  **{"maximum_cell_number": 2,
                                     "time_interval_threshold": 0.1})
            tic = time.time()
            for item in tqdm(test_set):
                pass
            toc = time.time()
            print(tam_type, ": ", test_set[0][0].shape, test_set[0][1], f"遍历时长: {toc - tic:.2f} s")
    elif test_num == 3:
        # 检查多个提取方法的提取效率
        dataset = "Closed_2tab"
        base_dir = get_filebase_dir()  # Closed_5tab regulator_Closed_2tab
        data_path = os.path.join(base_dir, dataset, "test.npz")
        X, y = load_data(data_path, max_load_time=80)
        for tam_type in ['ED3']:
            print(f"TAM_type is {tam_type}")
            test_set = TAMDataset(X, y, 5000, tam_type, 80, 1800, 100,
                                  **{
                                      "out_dt_thresholds": [ 0.0000, 0.1783, 0.4673, 2.5797, 9.7685, 44.4319],
                                      "in_dt_thresholds": [ 0.0000, 0.0105, 0.0391, 0.1011, 0.6342, 44.4345],
                                     })
            print(tam_type, ": ", test_set[0][0].shape, test_set[0][1])
            tic = time.time()
            for item in tqdm(test_set):
                pass
            toc = time.time()
            print(tam_type, ": ", test_set[0][0].shape, test_set[0][1], f"遍历时长: {toc - tic:.2f} s")
    elif test_num == 4:
        # 检查和比较多种提取方法的提取效率与平均耗时
        import tabulate
        datasets = ['Closed_2tab', 'wtfpad_Closed_2tab', 'front_Closed_2tab', 'regulator_Closed_2tab']
        tam_types = ['ED3', 'Mamba', 'EDRF', 'EDIAT']
        base_dir = get_filebase_dir()
        
        results_table = []
        
        for dataset in datasets:
            print(f"\n{'='*20} 测试数据集: {dataset} {'='*20}")
            data_path = os.path.join(base_dir, dataset, "test.npz")
            if not os.path.exists(data_path):
                print(f"数据文件不存在: {data_path}，跳过...")
                results_table.append([dataset] + ["N/A"] * len(tam_types))
                continue
                
            X, y = load_data(data_path, max_load_time=120)
            row_data = [dataset]
            
            for tam_type in tam_types:
                print(f"\nTAM_type 当前正在执行: {tam_type}")
                kwargs = {}
                if tam_type == 'Mamba':
                    kwargs = {"maximum_cell_number": 2, "time_interval_threshold": 0.1}
                elif tam_type in ['ED3', 'EDIAT']:
                    kwargs = {
                        "bin_count": 5,
                    }
                
                # 根据用户要求，加载时间为 120s，序列长度为 10000
                test_set = TAMDataset(X, y, 10000, tam_type, 120, 1800, 100, **kwargs)
                
                tic = time.time()
                for item in tqdm(test_set, desc=f"{dataset} - {tam_type}"):
                    pass
                toc = time.time()
                
                total_time = toc - tic
                n_samples = len(test_set)
                avg_time = total_time / n_samples if n_samples > 0 else 0
                
                print(f"[{dataset} - {tam_type}] -> 总遍历耗时: {total_time:.2f} s | 平均每条流量耗时: {avg_time*1000:.2f} ms | 总样本数: {n_samples}")
                
                # 在表格中同时展示总耗时和每条数据的平均耗时
                row_data.append(f"{total_time:.2f}s ({avg_time*1000:.2f}ms)")
                
            results_table.append(row_data)
            
        print(f"\n{'='*20} 测试结果汇总统计 {'='*20}")
        headers = ["Dataset"] + tam_types
        print(tabulate.tabulate(results_table, headers=headers, tablefmt="simple"))
    elif test_num == 5:
        dataset = "Closed_2tab"
        base_dir = get_filebase_dir()  # Closed_5tab regulator_Closed_2tab
        data_path = os.path.join(base_dir, dataset, "train.npz")
        X, y = load_data(data_path, max_load_time=80)
        print(len(X))