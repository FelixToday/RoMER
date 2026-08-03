import os
import sys
import pytz
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import time
import csv

class BaseLogger:
    def __init__(self, json_save_path: str, log_save_path: str, log_to_console: bool = True):
        """
        初始化训练日志记录器

        参数:
            save_path: 日志文件保存路径(.json)
            log_to_console: 是否在控制台输出日志信息
        """
        self.json_save_path = json_save_path
        self.log_save_path = log_save_path
        self.time_buf = {}
        self.outputs: Dict[str, List[Any]] = {}
        self.outputs["metadata"] = {
            'create_at': datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'),
            'update_at': None,
            'end_at': None
        }
        self.log_to_console = log_to_console

        # 配置日志格式
        self._configure_logging()

    def time_start(self, tag: str) -> None:
        """
        开始计时

        参数:
            tag: 计时标识符
        """
        self.time_buf[tag] = time.time()

    def time_end(self, tag: str) -> float:
        """
        结束计时并返回耗时(秒)

        参数:
            tag: 计时标识符

        返回:
            耗时(秒)
        """
        if tag not in self.time_buf:
            print(f"未找到计时标签: {tag}")

        duration = time.time() - self.time_buf[tag]
        del self.time_buf[tag]

        # 自动记录耗时到日志
        self._add_field(f"timing.{tag}", duration)
        self.info(f"{tag}:{duration:.2f} s")
        return duration

    def _configure_logging(self):
        """配置日志格式"""
        self.log_format = '%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s'
        self.date_format = '%Y-%m-%d %H:%M:%S,%f'

    def info(self, message: str, *args, is_logfile: bool = False, log_to_console: bool = True, **kwargs):
        """
        记录信息级别日志

        参数:
            message: 要记录的信息(可以使用%s等格式化占位符)
            *args: 格式化参数
            log_to_file: 是否写入日志
            **kwargs:
                - extra: 额外信息字典
        """
        # 获取调用信息
        frame = sys._getframe(1)
        filename = os.path.basename(frame.f_code.co_filename)
        func_name = frame.f_code.co_name
        lineno = frame.f_lineno

        # 格式化消息
        formatted_msg = message % args if args else message

        # 添加额外信息
        extra = kwargs.get('extra', {})
        if extra:
            formatted_msg += " " + " ".join(f"{k}={v}" for k, v in extra.items())

        # 格式化日志行
        log_entry = self.log_format % {
            'asctime': datetime.now(pytz.timezone('Asia/Shanghai')).strftime(self.date_format)[:-3],  # 去掉最后3位微秒
            'filename': filename,
            'funcName': func_name,
            'lineno': lineno,
            'levelname': 'INFO',
            'message': formatted_msg
        }

        # 控制台输出
        if self.log_to_console and log_to_console:
            print(log_entry)

        # 文件输出
        if is_logfile:
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(os.path.abspath(self.log_save_path)), exist_ok=True)
                mode = 'a' if os.path.exists(self.log_save_path) else 'w'
                with open(self.log_save_path, mode, encoding='utf-8') as f:
                    f.write(log_entry + '\n')
            except Exception as e:
                print(f"写入日志文件失败: {e}", file=sys.stderr)

    def _add_field(self, field_name: str, value: Optional[Any] = None, replace: bool = False) -> None:
        """
        添加或更新日志字段

        参数:
            field_name: 字段名称(支持点号分隔的嵌套字段，如'metrics.accuracy')
            value: 要记录的值(可选)
        """
        keys = field_name.split('.')

        local_outputs=self.outputs
        for key in keys[:-1]:
            if key not in local_outputs:
                local_outputs[key] = {}
            local_outputs=local_outputs[key]
        final_key = keys[-1]
        if final_key not in local_outputs or replace:
            local_outputs[final_key] = value
        else:
            if isinstance(local_outputs[final_key], list):
                local_outputs[final_key].append(value)
            else:
                local_outputs[final_key] = [local_outputs[final_key], value]

    def clear_field(self, field_name: str) -> None:
        """
        清除指定字段(支持点号分隔的嵌套字段，如'metrics.accuracy')

        参数:
            field_name: 要清除的字段名称
        """

        keys = field_name.split('.')
        assert keys[0] != 'metadata', '不能删除 metadata'
        current = self.outputs

        try:
            # 遍历到倒数第二个key
            for key in keys[:-1]:
                current = current[key]

            # 删除最后一个key
            final_key = keys[-1]
            if final_key in current:
                del current[final_key]
        except (KeyError, TypeError):
            # 如果路径不存在，则忽略
            pass

    def log(self, field_name: str, value: Any, replace: bool = False, unzip_dict: bool = False) -> None:
        """
        记录值到指定字段

        参数:
            field_name: 字段名称
            value: 要记录的值
        """
        if isinstance(value, dict) and unzip_dict:
            for k, v in value.items():
                self._add_field(f"{field_name}.{k}", v, replace=replace)
        else:
            self._add_field(field_name, value, replace)
        self.outputs['metadata']['update_at'] = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        self.save_to_json()

    def save_to_json(self) -> None:
        """
        保存日志到JSON文件(包含元数据)
        不使用缩进，生成紧凑的JSON格式
        """
        os.makedirs(os.path.dirname(os.path.abspath(self.json_save_path)), exist_ok=True)
        self.outputs['metadata']["end_at"] = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        duration_seconds = (
                datetime.strptime(self.outputs['metadata']["end_at"], '%Y-%m-%d %H:%M:%S') -
                datetime.strptime(self.outputs['metadata']["create_at"], '%Y-%m-%d %H:%M:%S')
        ).total_seconds()

        # 转换为 时:分:秒 格式
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        self.outputs['metadata']["duration"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        with open(self.json_save_path, 'w', encoding='utf-8') as f:
            json.dump(self.outputs, f, indent=4, ensure_ascii=False, separators=(',', ':'))

    def load_from_json(self) -> None:
        """
        从JSON文件加载日志数据
        """
        if os.path.exists(self.json_save_path):
            with open(self.json_save_path, 'r', encoding='utf-8') as f:
                self.outputs = json.load(f)


    def print_logs(self) -> None:
        """
        以美观格式打印当前日志内容
        """
        print(json.dumps(self.outputs, indent=4, ensure_ascii=False))
    def select_fields2csv(self, fields_list, savepath):

        def extract_list_of_dict(dict_list, prefix):
            """ 将 list[dict] 展开成多列，列名形如 prefix.key """
            result = {}
            if not dict_list:
                return result
            all_keys = set()
            for item in dict_list:
                if isinstance(item, dict):
                    all_keys.update(item.keys())
            for k in all_keys:
                col_name = f"{prefix}.{k}"
                col = []
                for item in dict_list:
                    if isinstance(item, dict):
                        col.append(item.get(k, ""))
                    else:
                        col.append("")
                result[col_name] = col
            return result

        def get_field_columns(field):
            """ 返回 dict: {col_name: col_values, ...} """
            # 支持 a.b 形式（显示请求子字段）
            if "." in field:
                parent, child = field.split(".", 1)
                parent_val = self.outputs.get(parent, "")

                # parent 是 list -> 从每个元素取 child
                if isinstance(parent_val, list):
                    col = []
                    for item in parent_val:
                        if isinstance(item, dict):
                            # 如果 child 里还包含点，则递归取值
                            if "." in child:
                                # 把 item 当作 outputs 临时对象处理 a.b.c 的剩余部分
                                # 简单实现：只支持一层点链（也可扩展）
                                sub_parent, sub_child = child.split(".", 1)
                                sub_val = item.get(sub_parent, "")
                                if isinstance(sub_val, dict):
                                    col.append(sub_val.get(sub_child, ""))
                                elif isinstance(sub_val, list):
                                    # 如果是 list，这里返回字符串以避免复杂展平冲突
                                    col.append(str(sub_val))
                                else:
                                    col.append(sub_val if sub_val is not None else "")
                            else:
                                col.append(item.get(child, ""))
                        else:
                            col.append("")
                    return {field: col}

                # parent 是 dict -> 直接取 child（作为单元）
                if isinstance(parent_val, dict):
                    # 如果 parent_val[child] 是 list of dict, 展开为多列
                    val = parent_val.get(child, "")
                    if isinstance(val, list) and all(isinstance(x, dict) for x in val):
                        return extract_list_of_dict(val, f"{parent}.{child}")
                    # 否则作为单列
                    if isinstance(val, list):
                        return {field: val}
                    return {field: [val]}

                return {field: [""]}

            # 非点字段（可能为 list / dict / 标量）
            value = self.outputs.get(field, "")

            # list of dict -> 展开成多列 field.subkey
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return extract_list_of_dict(value, field)

            # dict -> 检查 dict 内部是否有 list of dict，要进一步展开
            if isinstance(value, dict):
                nested_cols = {}
                for k, v in value.items():
                    # 如果 v 是 list of dict -> 展开为 field.k.subkey
                    if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                        nested_cols.update(extract_list_of_dict(v, f"{field}.{k}"))
                    # 如果 v 是普通 list -> 作为 field.k 一列
                    elif isinstance(v, list):
                        nested_cols[f"{field}.{k}"] = v
                    # 如果 v 是 dict -> 展开为 field.k.subkey（单元素列）
                    elif isinstance(v, dict):
                        for kk, vv in v.items():
                            nested_cols[f"{field}.{k}.{kk}"] = [vv]
                    else:
                        nested_cols[f"{field}.{k}"] = [v]
                if nested_cols:
                    return nested_cols
                # 若 dict 为空或没可展开项，退回为一个单列字符串表示
                return {field: [str(value)]}

            # 普通 list -> 直接作为一列
            if isinstance(value, list):
                return {field: value}

            # 标量 -> 单列
            return {field: [value]}

        # ---- collect columns in order of fields_list ----
        columns = {}
        col_order = []  # 保持列顺序
        for field in fields_list:
            col_dict = get_field_columns(field)
            for name, values in col_dict.items():
                # 如果名字已存在且你现在不想处理重复，只跳过或重写（这里选择重写）
                if name in columns:
                    # 你提到先不处理重复，这里用已有列覆盖新列（可按需改）
                    columns[name] = values
                else:
                    columns[name] = values
                    col_order.append(name)

        # ---- align lengths (用空字符串补齐) ----
        max_len = max((len(col) for col in columns.values()), default=0)
        for key in list(columns.keys()):
            col = columns[key]
            if len(col) < max_len:
                columns[key] = col + [""] * (max_len - len(col))

        # ---- write csv using col_order ----
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(col_order)
            for row in zip(*(columns[name] for name in col_order)):
                writer.writerow(row)


class ExperimentLogger:
    def __init__(
            self,
            json_path: Optional[Union[str, bytes]] = None,
            log_path: Optional[Union[str, bytes]] = None,
            is_console_out: bool = True
    ):
        """
        轻量级实验日志记录器

        Args:
            json_path: 数据(Metrics/Config)保存路径 (.json)
            log_path: 文本日志保存路径 (.txt/.log)
            is_console_out: 是否在控制台输出
        """
        self.json_path = json_path
        self.log_path = log_path
        self.console_out = is_console_out

        self._timers = {}
        # 初始化数据结构
        self.data: Dict[str, Any] = {
            "meta": {
                'created_at': datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S'),
                'updated_at': None,
                'duration': None
            }
        }

        # 日志格式配置
        self._log_fmt = '%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s'
        self._date_fmt = '%Y-%m-%d %H:%M:%S,%f'

        if not self.json_path:
            # 使用内置 print 防止递归调用
            import builtins
            builtins.print("[Info] json_path未设置，数据将不会保存到文件。")

    def start_timer(self, tag: str) -> None:
        """启动计时器"""
        self._timers[tag] = time.time()

    def stop_timer(self, tag: str) -> float:
        """停止计时器并自动 Record 耗时"""
        if tag not in self._timers:
            self.print(f"计时器 '{tag}' 未找到", level="WARN")
            return 0.0

        duration = time.time() - self._timers.pop(tag)

        # 自动记录
        self.record(f"timing.{tag}", duration)
        self.print(f"{tag} 耗时: {duration:.4f} s")
        return duration

    def print(self, msg: str, *args, save_to_file: bool = True, level: str = "INFO", **kwargs):
        """
        增强版打印函数：同时输出到控制台和日志文件

        Args:
            msg: 消息内容
            *args: 格式化参数 (e.g. "loss: %.4f", 0.5)
            save_to_file: 是否强制保存到文件
            level: 日志级别前缀
            **kwargs: extra={} 额外信息
        """
        # 1. 组装消息
        content = msg % args if args else msg
        if extra := kwargs.get('extra'):
            content += " " + " ".join(f"{k}={v}" for k, v in extra.items())

        # 2. 格式化日志行 (含时间、文件名、行号)
        frame = sys._getframe(1)
        log_entry = self._log_fmt % {
            'asctime': datetime.now(pytz.timezone('Asia/Shanghai')).strftime(self._date_fmt)[:-3],
            'filename': os.path.basename(frame.f_code.co_filename),
            'funcName': frame.f_code.co_name,
            'lineno': frame.f_lineno,
            'levelname': level,
            'message': "\n"+content
        }

        # 3. 控制台输出 (调用内置 print)
        if self.console_out:
            import builtins
            builtins.print(log_entry)

        # 4. 文件输出
        if self.log_path and save_to_file:
            try:
                self._ensure_dir(self.log_path)
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(log_entry + '\n')
            except Exception as e:
                import builtins
                builtins.print(f"写入日志失败: {e}", file=sys.stderr)

    def record(self, key: str, value: Any, overwrite: bool = False, unpack_dict: bool = False) -> None:
        """
        记录数据 (Record Data)

        Args:
            key: 字段名 (e.g. "train.loss")
            value: 值
            overwrite: 是否覆盖 (默认为追加到列表)
            unpack_dict: 是否展开字典
        """
        if isinstance(value, dict) and unpack_dict:
            for k, v in value.items():
                self._update_data_tree(f"{key}.{k}", v, overwrite)
        else:
            self._update_data_tree(key, value, overwrite)

        # 更新时间并保存
        self.data['meta']['updated_at'] = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        self._save_json()

    def _save_json(self) -> None:
        """保存数据到 JSON (内部调用)"""
        if not self.json_path:
            return

        # 计算时长
        start = datetime.strptime(self.data['meta']['created_at'], '%Y-%m-%d %H:%M:%S')
        sec = int((datetime.now(pytz.timezone('Asia/Shanghai')).replace(tzinfo=None) - start).total_seconds())
        self.data['meta']['duration'] = f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"

        try:
            self._ensure_dir(self.json_path)
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            import builtins
            builtins.print(f"JSON保存失败: {e}", file=sys.stderr)

    def load(self) -> None:
        """加载已有数据"""
        if self.json_path and os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

    def show_data(self) -> None:
        """打印当前记录的数据结构"""
        import builtins
        builtins.print(json.dumps(self.data, indent=4, ensure_ascii=False))

    def export_csv(self, fields: List[str], save_path: str) -> None:
        """导出指定字段到 CSV"""
        if not save_path: return

        columns = {}
        for field in fields:
            # 提取 + 扁平化
            raw = self._extract_value(self.data, field)
            columns.update(self._flatten_data(field, raw))

        if not columns: return

        # 对齐长度
        max_len = max((len(col) for col in columns.values()), default=0)
        for k in columns:
            columns[k] = list(columns[k]) + [""] * (max_len - len(columns[k]))

        try:
            self._ensure_dir(save_path)
            with open(save_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                headers = list(columns.keys())
                writer.writerow(headers)
                writer.writerows(zip(*(columns[h] for h in headers)))
            self.print(f"CSV已导出: {save_path}")
        except Exception as e:
            self.print(f"导出CSV错误: {e}", level="ERROR")

    # ================= 私有工具函数 =================

    def _ensure_dir(self, path: str):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _update_data_tree(self, key_path: str, value: Any, overwrite: bool):
        keys = key_path.split('.')
        curr = self.data
        for key in keys[:-1]:
            curr = curr.setdefault(key, {})
        last = keys[-1]
        if overwrite or last not in curr:
            curr[last] = value
        else:
            if isinstance(curr[last], list):
                curr[last].append(value)
            else:
                curr[last] = [curr[last], value]

    def _extract_value(self, data: Any, path: str) -> Any:
        if not path: return data
        keys = path.split('.')
        curr = data
        for i, key in enumerate(keys):
            if isinstance(curr, dict):
                curr = curr.get(key)
            elif isinstance(curr, list):
                rest = ".".join(keys[i:])
                return [self._extract_value(x, rest) for x in curr]
            else:
                return ""
            if curr is None: return ""
        return curr

    def _flatten_data(self, prefix: str, data: Any) -> Dict[str, List]:
        res = {}
        if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
            for k in set().union(*(d.keys() for d in data)):
                res[f"{prefix}.{k}"] = [x.get(k, "") for x in data]
        elif isinstance(data, dict):
            for k, v in data.items():
                res[f"{prefix}.{k}"] = v if isinstance(v, list) else [v]
        else:
            res[prefix] = data if isinstance(data, list) else [data]
        return res


# ================= 用法示例 =================
if __name__ == "__main__":
    # 初始化
    logger = ExperimentLogger("../cache/result.json", "../cache/log.txt")

    # 1. 使用 print 替代 info (更加直观)
    logger.print("任务开始...")

    # 2. 使用 record 替代 track (更有存档感)
    logger.record("config", {"lr": 0.01, "model": "ResNet"}, unpack_dict=True)

    logger.start_timer("train_process")

    # 模拟训练
    for i in range(3):
        logger.start_timer(f"step_{i}")
        time.sleep(0.1)

        # 记录指标
        logger.record("loss", 0.9 - i * 0.1)
        logger.record("accuracy", 0.1 + i * 0.2)

        logger.stop_timer(f"step_{i}")
        logger.print("Step %d 完成", i)

    logger.stop_timer("train_process")

    # 导出
    logger.export_csv(["loss", "accuracy", "timing"], "../cache/output.csv")


if __name__ == "__main__":
    logger = BaseLogger("../cache/wfa_result.json", "../cache/test.txt")
    logger.load_from_json()
    logger.print_logs()
    logger.select_fields2csv(["train.loss", "valid.result", "time"], "../cache/test.csv")
