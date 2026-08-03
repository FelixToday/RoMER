# -*- coding: utf-8 -*-

# @Author: Xianjun Li
# @E-mail: xjli@mail.hnust.edu.cn
# @Date: 2025/12/1 下午4:50
dataset_lib = {
    "Closed_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'CW_2tab', 'num_classes': 100},
    "Closed_3tab":{'num_tabs':3,'maximum_load_time': 120,'name':'CW_3tab', 'num_classes': 100},
    "Closed_4tab":{'num_tabs':4,'maximum_load_time': 120,'name':'CW_4tab', 'num_classes': 100},
    "Closed_5tab":{'num_tabs':5,'maximum_load_time': 120,'name':'CW_5tab', 'num_classes': 100},
    "wtfpad_Closed_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'CW_Pad', 'num_classes': 100},
    "front_Closed_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'CW_Fro', 'num_classes': 100},
    "regulator_Closed_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'CW_Reg', 'num_classes': 100},
    "Open_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'OW_2tab', 'num_classes': 101},
    "Open_3tab":{'num_tabs':3,'maximum_load_time': 120,'name':'OW_3tab', 'num_classes': 101},
    "Open_4tab":{'num_tabs':4,'maximum_load_time': 120,'name':'OW_4tab', 'num_classes': 101},
    "Open_5tab":{'num_tabs':5,'maximum_load_time': 120,'name':'OW_5tab', 'num_classes': 101},
    "wtfpad_Open_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'OW_Pad', 'num_classes': 101},
    "front_Open_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'OW_Fro', 'num_classes': 101},
    "regulator_Open_2tab":{'num_tabs':2,'maximum_load_time': 120,'name':'OW_Reg', 'num_classes': 101},
}


# ============================================================
# 服务器路径配置（可选）
# 如果你的机器 hostname 出现在下面的字典中，将使用对应的绝对路径；
# 否则回退到相对仓库根目录的默认路径（推荐，无需任何配置）。
# 也可在运行命令中显式指定 --file_base_dir / --checkpoint_path 覆盖。
# ============================================================
host_info = {
    # "your_hostname": {
    #     'host_name': 'my_server',
    #     'root_dir': '/path/to/repo',
    #     'filebase_dir': '/path/to/npz_dataset',
    # },
}

# 默认路径（相对 wfa_main/run 工作目录）
DEFAULT_ROOT_DIR = "../.."                # 仓库根目录
DEFAULT_FILEBASE_DIR = "../npz_dataset"   # npz 数据集根目录


def get_root_path():
    import socket
    info = host_info.get(socket.gethostname())
    return info['root_dir'] if info else DEFAULT_ROOT_DIR


def get_ckp_path():
    import socket, os
    info = host_info.get(socket.gethostname())
    root = info['root_dir'] if info else DEFAULT_ROOT_DIR
    return os.path.join(root, "checkpoints")


def get_host_name():
    import socket
    info = host_info.get(socket.gethostname())
    return info['host_name'] if info else "default"


def get_host_dataset_dir():
    return get_filebase_dir()


def get_filebase_dir():
    import socket
    info = host_info.get(socket.gethostname())
    return info['filebase_dir'] if info else DEFAULT_FILEBASE_DIR


def get_machine_name():
    import socket
    info = host_info.get(socket.gethostname())
    return info['host_name'] if info else "default"