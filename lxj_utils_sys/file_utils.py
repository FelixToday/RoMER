import os
import json
import csv
import datetime
import pytz
from pathlib import Path
from typing import Union, Any, List, Dict, Optional


def count_python_lines(
        target_paths: Union[str, Path, List[Union[str, Path]]],
        ignore_paths: Optional[Union[List, Any]] = None,
        save_path: Optional[str] = None,
        is_print: bool = True,
        view_mode: str = 'all'
) -> int:
    """
    基于用户提供的健壮逻辑重写行数统计功能。
    统计指定路径下所有 .py 文件的代码行数、空白行数、注释行数。
    
    Args:
        target_paths: 要统计的根路径（支持具体文件、目录，或它们的列表）
        ignore_paths: 需要忽略的文件或目录路径列表。
        save_path: 保存统计结果的文本文件路径，若为None则不保存。
        is_print: 是否在控制台打印每个文件的处理详情。
        view_mode: 统计视图，可选 'folder', 'file' 或 'all'（默认）。

    Returns:
        int: 所有未被忽略的 .py 文件的纯代码行数总和。
    """
    import fnmatch
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None
        if is_print:
            print("【配置提示】建议通过 pip install tabulate 安装该库以获得更好的表格打印效果。")

    if isinstance(target_paths, (list, tuple)):
        target_paths = [Path(p).resolve() for p in target_paths]
    else:
        target_paths = [Path(target_paths).resolve()]

    for p in target_paths[:]:
        if not p.exists():
            if is_print:
                print(f"【警告】路径不存在已跳过: {p}")
            target_paths.remove(p)

    # 处理忽略列表
    ignore_patterns = []
    if ignore_paths:
        if isinstance(ignore_paths, (list, tuple)):
            for item in ignore_paths:
                ignore_patterns.append(str(item))
        else:
            ignore_patterns.append(str(ignore_paths))

    default_ignores = [
        '.git', '.idea', '.vscode', '__pycache__', 'venv', 'env',
        'build', 'dist', 'migrations', 'node_modules', '*.pyc', '*.egg-info'
    ]
    ignore_patterns.extend(default_ignores)

    def _should_ignore(path_or_name: str) -> bool:
        name = os.path.basename(path_or_name)
        abs_p = os.path.abspath(path_or_name)
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(name, pattern): return True
            if pattern == abs_p: return True
            if os.path.isdir(pattern) and abs_p.startswith(pattern): return True
        return False

    def _shorten_path(p, max_len=50):
        if len(p) <= max_len:
            return p
        return p[:10] + "..." + p[-(max_len - 13):]

    file_details = {}
    processed_files = set()

    def _analyze_single_file(filepath):
        abs_filepath = os.path.abspath(filepath)
        if abs_filepath in processed_files: return None
        if _should_ignore(abs_filepath): return None
        processed_files.add(abs_filepath)
        
        stats = {'code': 0, 'comment': 0, 'blank': 0, 'total': 0}
        try:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                with open(filepath, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
        except (PermissionError, OSError):
            return None

        stats['total'] = len(lines)
        in_docstring = False
        docstring_marker = None

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_docstring:
                    stats['comment'] += 1
                else:
                    stats['blank'] += 1
                continue

            if not in_docstring:
                if stripped.startswith('#'):
                    stats['comment'] += 1
                elif '"""' in stripped and stripped.count('"""') % 2 != 0:
                    in_docstring = True
                    docstring_marker = '"""'
                    if stripped.startswith('"""'):
                        stats['comment'] += 1
                    else:
                        stats['code'] += 1
                elif "'''" in stripped and stripped.count("'''") % 2 != 0:
                    in_docstring = True
                    docstring_marker = "'''"
                    if stripped.startswith("'''"):
                        stats['comment'] += 1
                    else:
                        stats['code'] += 1
                else:
                    stats['code'] += 1
            else:
                stats['comment'] += 1
                if docstring_marker and docstring_marker in stripped:
                    in_docstring = False
                    docstring_marker = None
        return stats

    # 执行扫描
    for t_path in target_paths:
        if t_path.is_file():
            if str(t_path).endswith('.py'):
                res = _analyze_single_file(str(t_path))
                if res:
                    file_details[str(t_path)] = res
        else:
            for r, dirs, files in os.walk(str(t_path)):
                for d in dirs[:]:
                    if _should_ignore(os.path.join(r, d)): dirs.remove(d)
                for file in files:
                    if file.endswith('.py'):
                        res = _analyze_single_file(os.path.join(r, file))
                        if res:
                            file_details[os.path.join(r, file)] = res

    # 汇报统计
    if not file_details:
        if is_print:
            print("未找到任何有效代码文件。")
        return 0

    def _generate_table(mode, file_details):
        from collections import defaultdict
        t_stats = {'code': 0, 'comment': 0, 'blank': 0, 'total': 0}
        t_data = []
        base_dir = os.getcwd()
        if mode == 'folder':
            folder_stats = defaultdict(lambda: {'code': 0, 'comment': 0, 'blank': 0, 'total': 0})
            for fpath, s in file_details.items():
                folder = os.path.dirname(fpath)
                try:
                    display = os.path.relpath(folder, base_dir)
                    if display.startswith(".."): display = str(folder)
                    elif display == "": display = "."
                except ValueError:
                    display = str(folder)
                for k in ['code', 'comment', 'blank', 'total']:
                    folder_stats[display][k] += s[k]
                    t_stats[k] += s[k]
            sorted_items = sorted(folder_stats.items(), key=lambda x: x[1]['code'], reverse=True)
            for name, s in sorted_items:
                ratio = f"{s['code'] / s['total'] * 100:.1f}%" if s['total'] else "0%"
                display_name = _shorten_path(name)
                t_data.append([display_name, s['code'], s['comment'], s['blank'], s['total'], ratio])
        else:
            sorted_items = sorted(file_details.items(), key=lambda x: x[1]['code'], reverse=True)
            for fpath, s in sorted_items:
                try:
                    display = os.path.relpath(fpath, base_dir)
                except ValueError:
                    display = os.path.basename(fpath)
                for k in ['code', 'comment', 'blank', 'total']:
                    t_stats[k] += s[k]
                ratio = f"{s['code'] / s['total'] * 100:.1f}%" if s['total'] else "0%"
                display_name = _shorten_path(display)
                t_data.append([display_name, s['code'], s['comment'], s['blank'], s['total'], ratio])
                
        t_ratio = f"{t_stats['code'] / t_stats['total'] * 100:.1f}%" if t_stats['total'] else "0%"
        t_data.append(["TOTAL ALL", t_stats['code'], t_stats['comment'], t_stats['blank'], t_stats['total'], t_ratio])
        
        headers = ["Target", "Code", "Comments", "Blanks", "Total", "Code%"]
        if tabulate:
            table_str = tabulate(t_data, headers=headers, tablefmt="simple", stralign="center", numalign="center")
        else:
            table_lines = [f"{str(h):>15}" for h in headers]
            table_str = " | ".join(table_lines) + "\n"
            for row in t_data:
                table_str += " | ".join([f"{str(c):>15}" for c in row]) + "\n"
        return f">>>>>> 统计模式: {mode.upper()} VIEW <<<<<<\n{table_str}", t_stats

    # 构建并打印报告
    report_opts = []
    if isinstance(target_paths, (list, tuple)):
        report_opts.append(f"扫描目标: {len(target_paths)} 个输入路径")
    else:
        report_opts.append(f"扫描目标: {target_paths}")
    report_str = "\n".join(report_opts) + "\n\n"

    modes_to_run = ['folder', 'file'] if view_mode == 'all' else [view_mode]
    final_stats = {'code': 0}
    for m in modes_to_run:
        part_str, f_stats = _generate_table(m, file_details)
        report_str += part_str + "\n"
        final_stats = f_stats

    total_stats = final_stats

    if is_print:
        print(f"\n{report_str}\n")

    if save_path:
        try:
            output_dir = os.path.dirname(save_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("Python代码统计报告\n")
                f.write(f"统计时间: {datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                f.write(report_str + "\n")
                f.write("\n" + "=" * 70 + "\n")
            if is_print:
                print(f"详细报告已保存到: {save_path}")
        except Exception as e:
            if is_print:
                print(f"警告: 无法保存报告文件: {e}")

    return total_stats['code']


def _get_by_path(data: Union[Dict, List], path: str) -> Any:
    """内部通用函数：支持即使中间层是 list，也依然对每个元素取出深层次字典字段"""
    keys = path.split('.')
    cur = data

    for i, k in enumerate(keys):
        if isinstance(cur, list):
            rest_path = '.'.join(keys[i:])
            return [_get_by_path(item, rest_path) for item in cur]
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur


def _normalize_value(v: Any) -> List[Any]:
    """内部通用函数：将任意取出的值转化成列表作为 CSV 的列"""
    if isinstance(v, list):
        return v
    if isinstance(v, (int, float, str)):
        return [v]
    return [None]


def build_table(data: Union[Dict, List], paths: List[str]) -> List[Dict[str, Any]]:
    """根据给定的配置提取字段，构造二维表表示."""
    columns = {}
    max_len = 0

    for p in paths:
        v = _get_by_path(data, p)
        v = _normalize_value(v)
        columns[p] = v
        max_len = max(max_len, len(v))

    table = []
    for i in range(max_len):
        row = {}
        for p, col in columns.items():
            row[p] = col[i] if i < len(col) else None
        table.append(row)

    return table


def extract_json_to_csv(
        json_path_or_dict: Union[str, Dict],
        paths: List[str],
        output_csv_path: str = "./output.csv",
        verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    从 JSON 文件或字典中提取指定的深层属性字典/列表，转换为平面表格并输出到 CSV文件。

    Args:
        json_path_or_dict: json文件路径字符串，或者已加载的字典列表。
        paths: 要提取的值路径，用点语法表示 (例如 'epochs.train.loss')。
        output_csv_path: 写入的csv目标绝对或相对路径。
        verbose: 是否打印提示信息。

    Returns:
        包含表行数据的列的列表字典 (list[dict])。
    """
    if isinstance(json_path_or_dict, dict):
        data = json_path_or_dict
        if verbose:
            print("[INFO] 直接使用传入的 dict 数据")
    else:
        if verbose:
            print(f"[INFO] 从文件加载 JSON: {json_path_or_dict}")
        with open(json_path_or_dict, "r", encoding="utf-8") as f:
            data = json.load(f)

    if verbose:
        print(f"[INFO] 提取路径: {paths}")
    
    table = build_table(data, paths)

    if verbose:
        print(f"[INFO] 共提取 {len(table)} 行数据")

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=paths)
        writer.writeheader()
        writer.writerows(table)

    if verbose:
        print(f"[INFO] CSV 结果已写入: {output_csv_path}")

    return table


if __name__ == "__main__":
    # 示例调用
    pass
