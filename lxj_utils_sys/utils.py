import configparser
import json
from typing import Union
import time
import functools
from typing import Callable, Any
import numpy as np
def print_colored(text, color="", is_print=True):
    """
    输出带颜色的文字，并返回该字符串（带颜色）
    """
    colors = {
        "black": "30",
        "red": "91",
        "green": "92",
        "yellow": "93",
        "blue": "94",
        "magenta": "95",
        "cyan": "96",
        "white": "97",
    }
    if color == "" or color.lower() not in colors:
        str_info = text
    else:
        color_code = colors.get(color.lower(), "37")  # 默认白色
        str_info = f"\033[{color_code}m{text}\033[0m"
    if is_print:
        print(str_info)
    return str_info

def print_title(text, color="red", is_print=True):
    print_colored("="*20+" "+text+" "+"="*20, color, is_print)

def sort_lists(*lists):
    """
    将输入的列表按照第一个列表进行排序
    """
    # 获取第一个列表
    first_list = lists[0]
    # 对第一个列表的索引进行排序
    sorted_indices = sorted(range(len(first_list)), key=lambda i: first_list[i])
    # 根据排序后的索引重新排序所有列表
    sorted_lists = [[lst[i] for i in sorted_indices] for lst in lists]
    return sorted_lists

def str_to_bool(value):
    """
    将字符串值转换为布尔值。

    该函数接受一个字符串或布尔值输入，并将其转换为对应的布尔值。
    如果输入已经是布尔值，则直接返回。
    对于字符串输入，不区分大小写，支持多种常见布尔表示形式。

    Args:
        value: 待转换的值，可以是布尔值或字符串。
            如果是字符串，支持的真值包括：'yes', 'true', 't', 'y', '1'
            支持的假值包括：'no', 'false', 'f', 'n', '0'

    Returns:
        bool: 转换后的布尔值

    Raises:
        argparse.ArgumentTypeError: 当输入无法识别为有效的布尔值时抛出异常

    Examples:
        >>> str_to_bool('True')
        True
        >>> str_to_bool('0')
        False
        >>> str_to_bool(True)
        True

    Note:
        - 字符串比较不区分大小写
        - 如果输入值不在支持的范围内，会抛出ArgumentTypeError异常
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')
def same_seed(fix_seed):
    """
    设置随机种子
    """
    import torch.backends.cudnn as cudnn
    import random
    import torch
    import numpy as np

    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    rng = np.random.RandomState(fix_seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

def print_dict(d):
    import json
    print(json.dumps(d, indent=4, ensure_ascii=False))


class IncrementalMeanCalculator:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def add(self, new_value, skip_count=False):
        """
        添加新值。
        :param new_value: 可以是数值、列表、元组或 numpy 数组
        :param skip_count: 为 True 时，只累加数值，不增加计数（count）
        """
        # 统一转换为 numpy 数组处理，这样可以一次性处理多维、单值、列表等所有情况
        try:
            arr = np.array(new_value, dtype=float)
            if arr.size == 0:
                return

            # 累加总和
            self.total += np.sum(arr)

            # 如果不跳过计数，则增加 count
            if not skip_count:
                self.count += arr.size

        except (ValueError, TypeError):
            print(f"警告：包含无法转换的非数值类型: {new_value}")

    def get(self, round_num=None):
        if self.count == 0:
            return 0.0

        result = self.total / self.count
        return result if round_num is None else round(result, round_num)



def parse_args(parser, is_print_help=False):
    import configparser
    if isinstance(parser, configparser.ConfigParser):
        def parse_value(value):
            value = value.strip()
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    if value.lower() in ('true', 'false'):
                        return value.lower() == 'true'
                    return value
        args_data = {k: parse_value(v) for k, v in parser['config'].items()}
        help_info = {}
    else:
        args = parser.parse_args()
        args_data = vars(args)
        help_info = {}
        for action in parser._actions[1:]:
            key = action.dest
            if key not in args_data:
                continue
            else:
                help_info[key] = action.help
    if is_print_help:
        print_config_info(args_data, help_info)
    return args_data, help_info

def get_dict_structure(data: Union[dict, str]) -> None:
    """
    单个函数完成JSON结构分析和打印
    """
    # 处理输入数据
    if isinstance(data, str):
        with open(data, 'r', encoding='utf-8') as f:
            data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("输入必须是dict或者json文件路径")

    # 内部递归函数，用户不需要调用
    def _process_value(value, indent: int = 0):
        """处理值的递归函数"""
        space = "    " * indent

        if isinstance(value, dict):
            print(space + "{")
            for k, v in value.items():
                print(f"{space}    '{k}': ", end="")
                _process_value(v, indent + 1)
            print(space + "}")

        elif isinstance(value, list):
            if not value:
                print("list")
            else:
                elem_types = set(type(v) for v in value)
                if elem_types <= {int, float}:
                    print("list(float)")
                elif elem_types <= {str}:
                    print("list(str)")
                else:
                    print("list")

        elif isinstance(value, str):
            print("str")
        elif isinstance(value, int):
            print("int")
        elif isinstance(value, float):
            print("float")
        elif isinstance(value, bool):
            print("bool")
        elif value is None:
            print("NoneType")
        else:
            print(str(type(value)))

    # 开始处理
    _process_value(data)


def print_config_info(config: dict, config_description: dict, sorted_keys: bool = False):
    """
    打印配置信息表格

    参数:
    config: 配置字典，包含配置项和对应的值
    config_description: 描述字典，包含配置项的描述信息
    sorted_keys: 是否按键名排序输出，默认为 False
    case_sensitive: 排序时是否区分大小写，默认为 True
    """
    from tabulate import tabulate

    # 准备表格数据
    table_data = []

    # 根据 sorted_keys 参数决定是否排序
    if sorted_keys:
        keys_to_iterate = sorted(config.keys(), key=lambda x: x.lower())
    else:
        keys_to_iterate = config.keys()

    for key in keys_to_iterate:
        value = config[key]

        # 获取描述信息，如果不存在则使用"暂无"
        description = config_description.get(key, print_colored("暂无描述", color="red", is_print=False))

        # 处理特殊值（如None、列表、字典等）
        if value is None:
            value_str = "None"
        else:
            value_str = str(value)

        table_data.append([key, value_str, description])

    # 设置表格头
    headers = ["parameter name", "value", "description"]

    # 使用tabulate输出表格
    print("=" * 20 + "配置信息" + "=" * 20)
    print(tabulate(table_data, headers=headers, tablefmt="simple"))
    print("=" * 20 + "配置信息" + "=" * 20)

    # 可选：输出统计信息
    total_keys = len(config)
    described_keys = sum(1 for key in config if key in config_description)
    print(f"\n配置项总数: {total_keys}")
    print(f"已描述项: {described_keys}")
    print(f"未描述项: {total_keys - described_keys}")




def timer(func: Callable) -> Callable:
    """
    基础计时装饰器，用于测量函数执行时间

    使用示例：
        @timer
        def my_function():
            time.sleep(1)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()  # 使用高精度计时器
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print(f"函数 {func.__name__} 执行时间: {elapsed_time:.6f} 秒")
        return result

    return wrapper


def save_plot(save_path: str, dpi: int = 300) -> None:
    """
    将当前matplotlib图形保存为指定格式的图片

    参数:
        save_path (str): 图片保存路径（包含文件名和扩展名）
        dpi (int, 可选): 图片分辨率，默认为300

    支持格式:
        pdf, eps, svg, png, jpg

    示例:
        save_plot('output/figure.png', dpi=600)
    """
    import os
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("未安装 matplotlib，无法保存图片")
        return
        
    if not save_path:
        print("保存路径为空，图片未保存")
        return

    supported_formats = ['pdf', 'eps', 'svg', 'png', 'jpg']
    file_ext = save_path.split('.')[-1].lower()

    if file_ext not in supported_formats:
        raise ValueError(f"支持格式: {supported_formats}，当前格式：{file_ext}")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', format=file_ext, dpi=dpi)
    print(f"图片已保存至: {save_path}")





def get_func_use_dic_fields(input_funcs, target_id="args", ignore_fields=None):
    """
    鲁棒版：提取一个或多个函数中使用的字典字段
    :param input_funcs: 可以是单个函数，也可以是函数列表/元组
    :param target_id: 字典变量名
    :param ignore_fields: 需要手动过滤的字段
    """
    import ast
    import inspect
    from collections.abc import Iterable
    # 1. 统一输入格式：将单个函数转为列表
    if not isinstance(input_funcs, Iterable):
        funcs = [input_funcs]
    else:
        funcs = input_funcs

    ignore_fields = set(ignore_fields) if ignore_fields else set()
    all_used_fields = set()
    all_assigned_fields = set()

    for func in funcs:
        # 2. 增加类型检查，防止传入非函数对象
        if not callable(func):
            continue

        try:
            source = inspect.getsource(func)
            tree = ast.parse(source)
        except (Exception, TypeError) as e:
            # 某些内置函数或动态生成的函数可能无法获取源码
            print(f"跳过函数 {getattr(func, '__name__', 'Unknown')}: {e}")
            continue

        for node in ast.walk(tree):
            # A. 识别：args['key'] 或 args["key"]
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == target_id:
                    # 只有当 key 是常量字符串时才记录
                    field_name = None
                    if isinstance(node.slice, ast.Constant):  # Python 3.8+
                        field_name = node.slice.value
                    elif hasattr(node.slice, 'value') and isinstance(node.slice.value, ast.Constant):  # 旧版兼容
                        field_name = node.slice.value.value

                    if isinstance(field_name, str):
                        if isinstance(node.ctx, ast.Store):
                            all_assigned_fields.add(field_name)
                        else:
                            all_used_fields.add(field_name)

            # B. 识别：args.get('key')
            elif isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Attribute) and node.func.attr == 'get' and
                        isinstance(node.func.value, ast.Name) and node.func.value.id == target_id):
                    if node.args and isinstance(node.args[0], ast.Constant):
                        field_name = node.args[0].value
                        if isinstance(field_name, str):
                            all_used_fields.add(field_name)

    # 3. 计算最终字段：读取的 - 赋值的 - 忽略的
    final_fields = all_used_fields - all_assigned_fields - ignore_fields
    return sorted(list(final_fields))

if __name__ == '__main__':
    print_title("你好")