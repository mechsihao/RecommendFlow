import datetime
import os
import tensorflow as tf
from typing import List, Dict, Union, Any
import importlib
import numpy as np


def str2bool(input_str: str) -> bool:
    return True if input_str.lower() == 'true' else False


def type_trans_fun(type_fun: Union[str, Any], value: Any):
    type_map: Dict[str, type] = {
        "str": str, "int": int, "set": set, "list": list, "float": float,
        "dict": lambda x: {i.strip().split("=")[0]: "=".join(i.strip().split("=")[1:]) for i in x.strip().split(";")},
        "float32": np.float32, "float64": np.float64, "tensor": tf.constant
    }
    if isinstance(type_fun, str):
        if type_fun.lower() not in type_map:
            raise ValueError(f"type function: `{type_fun}` dose not supported")
        else:
            if type_fun.lower() == "dict":
                assert "=" in value, "Dict trans failed, context will be splited by `=` when you set type_fun=`dict`"
            return type_map[type_fun.lower()](value)
    else:
        return type_fun(value)


def str2list(input_str: str, sep: str = ",", trans_type: Union[type, str] = str) -> List[str]:
    return [type_trans_fun(trans_type, i.strip()) for i in input_str.split(sep) if i.strip()]


def str2dict(input_str: str, trans_type: Union[type, str] = str) -> Dict[str, str]:
    """
    :param input_str: 格式为 a=1;b=2;c=3
    :param trans_type: value_type 类型
    :return: Dict[str,value]
    """
    res = {}
    for params in input_str.strip().split(";"):
        key, value = params.strip().split("=")
        res[key.strip()] = type_trans_fun(trans_type, value.strip())
    return res


def str2scale(input_str: str):
    """返回一个范围函数，如果x不在该范围内就return False，否则为True
    """
    left, right = input_str.strip().split(",")
    left_side, right_side = left.strip()[0], right.strip()[-1]
    left_num, right_num = float(left.strip()[1:]), float(right.strip()[:-1])

    if left_side == "[" and right_side == "]":
        return lambda x: left_num <= x <= right_num
    elif left_side == "[" and right_side == ")":
        return lambda x: left_num <= x < right_num
    elif left_side == "(" and right_side == "]":
        return lambda x: left_num < x <= right_num
    elif left_side == "(" and right_side == ")":
        return lambda x: left_num < x < right_num
    else:
        raise ValueError(f"Unknown left_side={left_side} or right_side={right_side}")


def str2debug(x):
    return x.lower() in ("test", "debug")


def str2dayno(x: str, mode: str = "patten") -> Union[str, List[str]]:
    """
    解析字符串，输出dayno list
    :param x: 时间字符串
    :param mode: 模式。支持两种：list、patten, 其中patten模式下返回linux可支持的通配符

    :Examples
        格式有如下几种情况
        - 1.直接输入前后端, '['代表闭区间, '('代表开区间，无符号则代表闭区间：
            - 例1 [20221126~20221128) = ['20221126', '20221127']
            - 例2 20221126~20221128 = ['20221126', '20221127', '20221128']
            - 例3 20221128 = ['20221128']
        - 2.加减式：
            - 'YYYYMMDD+x'/'YYYYMMDD-x' 代表从YYYYMMDD加/减x天，dayno_list长度为x+1，包含该天，例1 '20221128-2' =  ['20221126', '20221127', '20221128']
            - 'YYYYMMDD+:x'/'YYYYMMDD-:x' 代表从YYYYMMDD加/减x天，dayno_list长度为x，不包含该天，例1 '20221128-:2' =  ['20221126', '20221127']
    :return:
    """
    if "~" in x:
        left, right = x.split("~")
        left_symbol, right_symbol = left[0] if len(left) == 9 else "[", right[-1] if len(right) == 9 else "]"
        left_dayno, right_dayno = left[1:] if len(left) == 9 else left, right[:-1] if len(right) == 9 else right
        left_dayno, right_dayno = datetime.datetime.strptime(left_dayno, "%Y%m%d"), datetime.datetime.strptime(right_dayno, "%Y%m%d")
        dayno_list = [(left_dayno + datetime.timedelta(days=i)).strftime("%Y%m%d") for i in range((right_dayno - left_dayno).days + 1)]
        if left_symbol == "(":
            dayno_list.pop(0)
        if right_symbol == ")":
            dayno_list.pop()
    elif "+:" in x or "-:" in x:
        opt, direct = "+:" if "+:" in x else "-:", 1 if "+:" in x else -1
        base_dayno, diff_nums = datetime.datetime.strptime(x.split(opt)[0], "%Y%m%d"), int(x.split(opt)[-1])
        dayno_list = [(base_dayno + datetime.timedelta(days=direct * i)).strftime("%Y%m%d") for i in range(1, diff_nums + 1)]
    elif "+" in x or "-" in x:
        opt, direct = "+" if "+" in x else "-", 1 if "+" in x else -1
        base_dayno, diff_nums = datetime.datetime.strptime(x.split(opt)[0], "%Y%m%d"), int(x.split(opt)[-1])
        dayno_list = [(base_dayno + datetime.timedelta(days=direct * i)).strftime("%Y%m%d") for i in range(diff_nums + 1)]
    elif len(x) == 8:
        dayno_list = [x]
    else:
        raise Exception(f"Unknown input='{x}'")

    if mode == "list":
        return dayno_list
    elif mode == "patten":
        prefix = os.path.commonprefix(dayno_list)
        patten = prefix + "{" + ",".join([dayno[len(prefix):] for dayno in sorted(dayno_list)]) + "}"
        return patten
    else:
        raise Exception(f"Unknown mode='{mode}'")


def make_simplified_name_function(name: str):
    return "".join([i[0] for i in name.split("_")])


def str2loss(loss_name: str) -> List[str]:
    """
    通过字符串指定loss，新增loss最好使用tf.function加速
    :param loss_name: 可以是字典中的key和value，如果新增loss，最好在这里注册下，方便使用
    :return: 返回loss函数本身
    """
    some_pkgs = ["tf", "K", "keras", "match_losses", "loss", "losses"]
    package, loss = ".".join(loss_name.split(".")[:-1]), loss_name.split(".")[-1]
    if package == "":
        raise ValueError(f"Expect loss str is 'package.class', got {loss}, please given full package route.")
    else:
        losses = importlib.import_module(package)

    support_class = [i for i in losses.__dir__() if not i.startswith("__") and i not in some_pkgs]
    support_dict = {make_simplified_name_function(i): i for i in support_class}
    try:
        loss = getattr(losses, support_dict[loss]) if loss in support_dict else getattr(losses, loss)
    except AttributeError as e:
        info = ',\n    '.join(sorted(support_class + list(support_dict.keys())))
        raise AttributeError(f"""Load Loss Error\nSupport Loss :\n[\n    {info}\n]\n Detail: {e}""")
    return loss
