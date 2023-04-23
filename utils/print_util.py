import json
from argparse import Namespace
from typing import Dict, Any

import numpy as np
from pandas import DataFrame, Series


def print_line(line, max_len, prefix="", sep="", suffix="", gap=""):
    if line[0][0] == "=":
        sep = "=" * len(sep)
    s = prefix + gap + sep.join([f"{' ' * (max_len - len(i))}{i}" for i in line]) + gap + suffix
    print(s)


def print_metric(metric: DataFrame):
    """
    打印模型评估指标
    :param metric: 输入的矩阵，第一列必须为指标名称
    """
    metric_name_col = metric.columns[0]
    metric[metric_name_col] = metric[metric_name_col].astype(str)

    for col in metric.columns[1:]:
        metric[col] = metric[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else ','.join(x) if isinstance(x, list) else str(x))

    max_len = max([len(i) for i in metric.values.reshape(-1)]) + 2
    l, w = metric.shape

    print_line(["═" * max_len] * w, max_len, prefix="╔", sep="╦", suffix="╗", gap="")
    print_line([i.upper() for i in metric.columns], max_len, prefix="║", sep="║", suffix="║", gap="")
    print_line(["═" * max_len] * w, max_len, prefix="╠", sep="╬", suffix="╣", gap="")
    for i, line in enumerate(metric.values):
        print_line(line, max_len, prefix="║", sep="║", suffix="║", gap="")
    print_line(["═" * max_len] * w, max_len, prefix="╚", sep="╩", suffix="╝", gap="")


def print_mat(name, mat):
    print(name)
    shape = mat.shape
    if len(shape) == 1:
        print(mat)
    else:
        for line in mat:
            print(np.around(line, 4))
    print()


def print_metrix(mat_name: str, input_metrix: np.array):
    """
    打印矩阵
    :param mat_name: 名称
    :param input_metrix: 输入的矩阵
    """
    metrix = np.array([f"{i:.4f}" for i in input_metrix.reshape(-1)]).reshape(input_metrix.shape)
    if len(metrix.shape) == 1:
        p_metrix = metrix[None, :]
        l, w = p_metrix.shape
    elif len(metrix.shape) == 2:
        p_metrix = metrix
        l, w = p_metrix.shape
    else:
        d_end = metrix.shape[-1]
        d_end_2 = metrix.shape[-2]
        t_metrix = metrix.reshape(-1, d_end)
        l, w = t_metrix.shape
        max_len = max([max([len(i) for i in line]) for line in t_metrix])
        res = []
        for i, line in enumerate(t_metrix):
            if i % d_end_2 == 0 and i != 0 and i != l:
                res.append(["=" * max_len] * w)
            res.append(line)
        p_metrix = np.array(res)

    max_len = max([len(i) for i in p_metrix.reshape(-1)])
    print(mat_name)
    for line in p_metrix:
        print_line(line, max_len, prefix="[", sep=", ", suffix="]", gap="")
    print()


def print_series(name, series: Series) -> None:
    series_dict = series.to_dict()
    keys = list(series_dict.keys())
    values = list(series_dict.values())
    max_key_len = max(max([len(str(i)) for i in keys]), 5)
    max_value_len = max(max([len(str(i)) for i in values]), 8)
    print(f"{name}")
    print("┏" + "━" * (max_key_len + 2) + "┳" + "━" * (max_value_len + 2) + "┓")
    print("┃" + " " * (max_key_len - 2) + "KEY ┃ VALUE" + " " * (max_value_len - 4) + "┃")
    print("┣" + "━" * (max_key_len + 2) + "╋" + "━" * (max_value_len + 2) + "┫")
    for key, value in zip(keys, values):
        print("┃ " + " " * (max_key_len - len(str(key))) + f"{key} ┃ {value}" + " " * (max_value_len - len(str(value))) + " ┃")
    print("┗" + "━" * (max_key_len + 2) + "┻" + "━" * (max_value_len + 2) + "┛")


def print_dataframe_line(lines, index, max_index_len, max_key_lens, start, sep, end, blank):
    for i in range(len(lines) + 1):
        content = lines[i - 1] if i >= 1 else " "
        if i == 0:
            print(start + blank + index + blank * (max_index_len - len(index) + 1), end="")
        elif i == len(lines):
            print(sep + blank + content + blank * (max_key_lens[i - 1] - len(content) + 1) + end)
        else:
            print(sep + blank + content + blank * (max_key_lens[i - 1] - len(content) + 1), end="")


def print_dataframe(name: str, input_df: DataFrame) -> None:
    df = input_df[:]
    cols = df.columns

    for col in cols:
        if col == "count":
            df[col] = df[col].apply(lambda x: int(x))
        else:
            df[col] = df[col].apply(lambda x: f"{x:.5f}" if isinstance(x, float) else x)

    index = "INDEX"
    max_index_len = max(max([len(str(i)) for i in df.index.values]), len(index))
    max_key_lens = [max(max([len(str(i)) for i in df[col].values]), len(col)) for col in cols]

    print(f"{name}")
    print_dataframe_line(["━"] * len(cols), "━", max_index_len, max_key_lens, "┏", "┳", "┓", "━")
    print_dataframe_line(cols, index, max_index_len, max_key_lens, "┃", "┃", "┃", " ")
    print_dataframe_line(["━"] * len(cols), "━", max_index_len, max_key_lens, "┣", "╋", "┫", "━")

    for index, line in df.to_dict("index").items():
        print_dataframe_line([str(i) for i in line.values()], str(index), max_index_len, max_key_lens, "┃", "┃", "┃", " ")

    print_dataframe_line(["━"] * len(cols), "━", max_index_len, max_key_lens, "┗", "┻", "┛", "━")


def print_args_info(args_dict: Dict[str, str]):
    """
    用于打印Dict类型输入的参数
    :param args_dict:
    :return:
    """
    keys = list(args_dict.keys())
    values = list(args_dict.values())
    print("Args Comments:")

    def get_type_value(v: str):
        if "[" in v and "]" in v:
            return v[v.index("["): v.index("]")].strip("[").strip("]").strip(), v[v.index("]") + 1:].strip()
        else:
            return str(type(v)).split("'")[1].strip(), v

    type_max_len = max([len(get_type_value(i)[0]) for i in values]) + 2
    get_type_str = lambda x: str(" " * ((type_max_len - len(x) + 1) // 2) + x + " " * ((type_max_len - len(x)) // 2))[:type_max_len]

    max_key_len = max([len(i) for i in keys])
    max_value_len = max([len(get_type_value(str(i))[1]) for i in values])

    print("┏" + "━" * (max_key_len + 2) + "┳" + "━" * type_max_len + "┳" + "━" * (max_value_len + 2) + "┓")
    print("┃" + " " * (max_key_len - 4) + "ARG   ┃" + get_type_str("TYPE") + "┃   COMMENTS" + " " * (max_value_len - 9) + "┃")
    print("┣" + "━" * (max_key_len + 2) + "╋" + "━" * type_max_len + "╋" + "━" * (max_value_len + 2) + "┫")
    for key, value in zip(keys, values):
        t, value = get_type_value(value)
        print("┃ " + " " * (max_key_len - len(key)) + f"{key} ┃" + get_type_str(t) + f"┃ {str(value)}" + " " * (
                max_value_len - len(str(value))
        ) + " ┃")
    print("┗" + "━" * (max_key_len + 2) + "┻" + "━" * type_max_len + "┻" + "━" * (max_value_len + 2) + "┛")


def get_args_table_str(args_dict: Dict[str, Any], show_type: bool = True) -> str:
    keys = list(args_dict.keys())
    values = [i.__name__ if callable(i) else i for i in list(args_dict.values())]
    types = {k: str(type(v)).split("'")[1].strip() for k, v in args_dict.items()}

    max_key_len = max(max([len(i) for i in keys]), 10)
    max_value_len = max(max([len(str(i)) for i in values]), 15)
    ret = []
    print("Args Table:")
    if show_type:
        type_max_len = 6
        get_type_str = lambda tp: str(" " * ((type_max_len - len(tp) + 1) // 2) + tp + " " * ((type_max_len - len(tp)) // 2))[:type_max_len]
        ret.append("┏" + "━" * (max_key_len + 2) + "┳" + "━" * type_max_len + "┳" + "━" * (max_value_len + 2) + "┓")
        ret.append("┃" + " " * (max_key_len - 4) + "ARG   ┃" + get_type_str("TYPE") + "┃   VALUE" + " " * (max_value_len - 6) + "┃")
        ret.append("┣" + "━" * (max_key_len + 2) + "╋" + "━" * type_max_len + "╋" + "━" * (max_value_len + 2) + "┫")
        for key, value in zip(keys, values):
            type_str = types[key]
            t = type_str if type_str != "function" else "func"
            ret.append("┃ " + " " * (max_key_len - len(key)) + f"{key} ┃" + get_type_str(t) + f"┃ {str(value)}" + " " * (
                    max_value_len - len(str(value))
            ) + " ┃")
        ret.append("┗" + "━" * (max_key_len + 2) + "┻" + "━" * type_max_len + "┻" + "━" * (max_value_len + 2) + "┛")
    else:
        ret.append("┏" + "━" * (max_key_len + 2) + "┳" + "━" * (max_value_len + 2) + "┓")
        ret.append("┃" + " " * (max_key_len - 4) + "ARG   ┃   VALUE" + " " * (max_value_len - 6) + "┃")
        ret.append("┣" + "━" * (max_key_len + 2) + "╋" + "━" * (max_value_len + 2) + "┫")
        for key, value in zip(keys, values):
            ret.append("┃ " + " " * (max_key_len - len(key)) + f"{key} ┃ {value}" + " " * (max_value_len - len(str(value))) + " ┃")
        ret.append("┗" + "━" * (max_key_len + 2) + "┻" + "━" * (max_value_len + 2) + "┛")
    return "\n".join(ret)


def print_args_table(args: Namespace, show_type: bool = True):
    """
    用于打印argparse的输入参数
    :param args:
    :param show_type:
    :return:
    """
    print(get_args_table_str(args.__dict__, show_type))


def print_json(args_dict: Dict[str, Any]):
    res = json.dumps(args_dict, indent=2, ensure_ascii=False)
    print(res)
