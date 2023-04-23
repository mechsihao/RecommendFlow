import os
import time
import json
import random
import psutil
import datetime
import numpy as np
import pandas as pd
import urllib.request as urllib_req

from pandas import DataFrame
from typing import List, Union, Any

from utils.hdfs_util import ls_hdfs_paths
from utils.str_parser import str2scale
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow_io as tfio

os.environ["TFIO_VERSION"] = str(tfio.version)


def check_increase(new: float, base: float, scare_str: str):
    """
    判断 (new - base) / base 是否满足 scare_str 所设置的范围， scare_str 用法可查看 str2scale
    :param new: 新值
    :param base: 基准值
    :param scare_str: 范围字符串，[/]代表闭区间 (/)代表开区间
    :return: bool, 是否在该范围内
    """
    rate = (new - base) / base
    flag = str2scale(scare_str)((new - base) / base)
    msg = f"[Increase Info] 提升比例为{rate/100:.4f}%, {'' if flag else '不'}满足条件{'' if flag else ', 请检查！'}"
    return flag, msg


def l2_normalize(vecs):
    """l2标准化
    """
    norms = (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def send_msg(post_url, values):
    try:
        params = json.dumps(values).encode('utf8')
        headers = {'Accept-Charset': 'utf-8', 'Content-Type': 'application/json'}
        req = urllib_req.Request(url=post_url, data=params, headers=headers)
        rsp = urllib_req.urlopen(req)

        rsp_data = rsp.read()
        rsp.close()
        print("rsp_data=%s" % rsp_data)
        data_dict = json.loads(rsp_data)
        if 'ret' in data_dict:
            return data_dict['ret']
    except Exception as e:
        print(f"except,msg={e}")
    return -1


def send_tt_msg_once(tt_msg, to_user_list: List[str]):
    post_url = 'http://msg.ads.oppo.local/api/tt_push'  # 正式环境
    tt_msg = tt_msg
    values = {
        "msg": tt_msg,
        "to_user_list": to_user_list,
        "bizAlarm": "oppo-it-bp"
    }
    print('send tt msg:%s' % tt_msg)
    return send_msg(post_url, values)


def send_tt_msg(all_msg, to_user_list):
    # 发送TT告警
    for i in range(3):  # 重试3次
        if send_tt_msg_once(all_msg, to_user_list) == 0:
            break
        time.sleep(5)


def filter_illegal_chars(x: str):
    illegal_chars = """ !"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~、:，。、【】“”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥?…！，"""  # 包含空格
    for i in illegal_chars:
        x = x.replace(i, "")
    return x


def sample_neg_app(app_neg_weight, pos_app_list, neg_sample_nums):
    pos_app_set = set(pos_app_list)

    app_neg_name_list = []
    app_neg_weight_list = []
    for k, v in app_neg_weight.items():
        if k not in pos_app_set:
            # 这里是一种策略，后期可修改成不在pos_app_set里
            app_neg_name_list.append(k)
            app_neg_weight_list.append(v)

    return random.choices(app_neg_name_list, weights=app_neg_weight_list, k=len(pos_app_list) * neg_sample_nums)


def get_datetime(add_day: int = 0, fmt: str = '%Y.%m.%d-%H:%M:%S'):
    """
    获取日期，可自由配置格式：
        - %a 星期几的简写
        - %A 星期几的全称
        - %b 月分的简写
        - %B 月份的全称
        - %c 标准的日期的时间串
        - %C 年份后两位数字
        - %d 十进制表示的每月的第几天
        - %D 月/天/年
        - %e 在两字符域中，十进制表示的每月的第几天
        - %F 年-月-日
        - %g 年份的后两位数字，使用基于周的年
        - %G 年分，使用基于周的年
        - %h 简写的月份名
        - %H 24小时制的小时
        - %I 12小时制的小时
        - %j 十进制表示的每年的第几天
        - %m 十进制表示的月份
        - %M 十时制表示的分钟数
        - %n 新行符
        - %p 本地的AM或PM的等价显示
        - %r 12小时的时间
        - %R 显示小时和分钟：hh:mm
        - %S 十进制的秒数
        - %t 水平制表符
        - %T 显示时分秒：hh:mm:ss
        - %u 每周的第几天，星期一为第一天 （值从0到6，星期一为0）
        - %U 第年的第几周，把星期日做为第一天（值从0到53）
        - %V 每年的第几周，使用基于周的年
        - %w 十进制表示的星期几（值从0到6，星期天为0）
        - %W 每年的第几周，把星期一做为第一天（值从0到53）
        - %x 标准的日期串
        - %X 标准的时间串
        - %y 不带世纪的十进制年份（值从0到99）
        - %Y 带世纪部分的十制年份
        - %z，%Z 时区名称，如果不能得到时区名称则返回空字符。
        - %% 百分号
    :param add_day 相比于今天的变化天数，可正可负
    :param fmt 输出格式
    """
    return (datetime.datetime.today() + datetime.timedelta(days=add_day)).strftime(fmt)


def get_delta_seconds(start_time: str, end_time: str, fmt: str = '%Y.%m.%d-%H:%M:%S'):
    """
    获取时间间隔，可自由配置格式
    :param start_time 时间格式的字符串结束时间
    :param end_time 时间格式的字符串起始时间
    :param fmt 输出格式
    """
    delta = datetime.datetime.strptime(start_time, fmt) - datetime.datetime.strptime(end_time, fmt)
    return abs(delta.total_seconds())


def dump_csv(df: DataFrame, path: str, sep: str = "\t", index: bool = False, header: Union[bool, List[str]] = True, show: int = 0):
    sample_nums = len(df)
    df.to_csv(path, index=index, sep=sep, header=header)
    print(f"{path}保存成功")
    print(f"文件数量：{sample_nums}")
    print(f"列名：{df.columns}\n")
    if show > 0:
        print("数据采样展示：")
        print(df.sample(show))


def adapt_df_for_devices(device_nums: int, data: DataFrame) -> DataFrame:
    if device_nums != 1 and len(data) % device_nums != 0:
        org_nums = len(data)
        nums = org_nums // device_nums * device_nums
        print(f"[Warning]当前使用单机多卡策略, 数据量为{org_nums}无法整除, 已对数据进行裁切至{nums}")
        return data.iloc[:nums]
    else:
        return data


def check_and_make_cache_root(path: str = "__datacache__"):
    cache_root = os.path.join(os.getcwd(), path)
    if not os.path.exists(cache_root):
        os.makedirs(cache_root, exist_ok=True)
    return cache_root


def read_text_from_hdfs(path, names=None, sep="\t", usecols=None, error_bad_lines=False, nrows=None):
    file, names, col_nums, use_cols = [], names or [], len(names) if names else 0, usecols if usecols else list(range(len(names)))
    with tf.io.gfile.GFile(path, 'r') as gf:
        for line in gf:
            row = line.strip().rsplit(sep)
            if col_nums == 0:
                names, col_nums = row, len(row)
                if col_nums == 0:
                    raise Exception(f"First line data must be col names when `columns` was not given, got {names}, please check!")
            else:
                if len(row) == col_nums:
                    file.append(np.array(row)[use_cols].tolist())  # 只挑选 需要的 列
                elif error_bad_lines:
                    raise Exception(f"Col nums is {col_nums}, got {len(row)} data")
                else:
                    print(f"[IO WARNING] Error data {row[:2] + ['...']} was skipped.")

                if nrows and 0 <= nrows == len(file):
                    break
    return pd.DataFrame(file, columns=np.array(names)[use_cols].tolist())


def read_csv(path: str,
             sep: str = "\t",
             columns: List[str] = None,
             use_cols: List[int] = None,
             na: str = "-1",
             cache_data: bool = False,
             nrows: int = None) -> DataFrame:
    """读取csv文件，支持从HDFS读取csv文件，读取的文件必须有表头，否则会报错！
    """
    nrows = None if nrows and nrows < 0 else nrows
    start = time.time()
    cache_root = check_and_make_cache_root()

    if path.startswith("hdfs://") and cache_data:
        # 缓存文件仅当天有效，过时则重新生下载生成
        file = os.path.basename(path)
        cache_file = os.path.join(cache_root, file + f".cache.{get_datetime(fmt='%Y%m%d')}")
    else:
        cache_file = ""

    if not path.startswith("hdfs://"):
        print(f"Read data from local: {path}")
        df = pd.read_csv(path, sep=sep, dtype=str, names=columns, usecols=use_cols, nrows=nrows).fillna(na)
    else:
        if cache_data and os.path.exists(cache_file):
            print(f"Read cached data from local: {cache_file}")
            # 这样做的目的是，下次以同样传参调用接口，传入同样的columns不会报错，做到在hdfs端和本地读取无感的效果
            columns = np.array(columns)[use_cols].tolist() if columns and use_cols else columns
            df = pd.read_csv(cache_file, sep=sep, dtype=str, names=columns, nrows=nrows).fillna(na)
        else:
            print(f"Read data from hdfs: {path}")
            res = []
            total_rows = 0
            for path in ls_hdfs_paths(path):
                tmp_res = read_text_from_hdfs(path, names=columns, usecols=use_cols, sep=sep, nrows=nrows)
                total_rows += len(tmp_res)
                if nrows and total_rows > nrows:
                    res.append(tmp_res.head(total_rows - nrows))
                    break
                else:
                    res.append(tmp_res)
            df = pd.concat(res).reset_index(drop=True)
            print(f"Read record nums: {total_rows}")
            if cache_data and not nrows and not os.path.exists(cache_file):
                # 这里不允许指定nrows的时候缓存数据，否则 就会出现刚开始用nrows读一小部分数据，这时候一旦数据缓存，下一次想读全量数据就无法读取，一直会读取缓存的小部分数据
                header = False if columns else True
                df.to_csv(cache_file, sep=sep, index=False, header=header)
                print(f"Data has been cached to local: {cache_file}")

    print(f"DataFrame columns: {list(df.columns)}")
    print(f"Read dataFrame cost: {time.time() - start:.4f}s")
    return df


def save_text(contents: Union[Any, List[Any]], path: str):
    """
    以文本形式存储文件
    :param contents:
    :param path:
    :return:
    """

    if isinstance(contents, list):
        pass
    else:
        contents = [contents]

    with open(path, 'w') as f:
        for line in contents:
            f.write(str(line) + "\n")

    print(f"Text file saved to {path}")


def get_dataframe_line_str(lines, index, max_index_len, max_key_lens, start, sep, end, blank):
    res = ""
    for i in range(len(lines) + 1):
        content = lines[i - 1] if i >= 1 else " "
        if i == 0:
            res += start + blank + index + blank * (max_index_len - len(index) + 1)
        elif i == len(lines):
            res += sep + blank + content + blank * (max_key_lens[i - 1] - len(content) + 1) + end
        else:
            res += sep + blank + content + blank * (max_key_lens[i - 1] - len(content) + 1)
    return res


def df2str(input_df: DataFrame) -> str:
    """
    将df转换为str
    :param input_df:
    :return:
    """
    df = input_df[:]
    cols = df.columns

    res = []
    for col in cols:
        if col == "count":
            df[col] = df[col].apply(lambda x: int(x))
        else:
            df[col] = df[col].apply(lambda x: f"{x:.5f}" if isinstance(x, float) else x)

    index = "INDEX"
    max_index_len = max(max([len(str(i)) for i in df.index.values]), len(index))
    max_key_lens = [max(max([len(str(i)) for i in df[col].values]), len(col)) for col in cols]

    res.append(get_dataframe_line_str(["━"] * len(cols), "━", max_index_len, max_key_lens, "┏", "┳", "┓", "━"))
    res.append(get_dataframe_line_str(cols, index, max_index_len, max_key_lens, "┃", "┃", "┃", " "))
    res.append(get_dataframe_line_str(["━"] * len(cols), "━", max_index_len, max_key_lens, "┣", "╋", "┫", "━"))

    for index, line in df.to_dict("index").items():
        res.append(get_dataframe_line_str([str(i) for i in line.values()], str(index), max_index_len, max_key_lens, "┃", "┃", "┃", " "))

    res.append(get_dataframe_line_str(["━"] * len(cols), "━", max_index_len, max_key_lens, "┗", "┻", "┛", "━"))
    return "\n".join(res)


def men_percentage():
    return f"{psutil.virtual_memory().percent:.2f}%"


def split_and_shuffle(df: DataFrame, test_size: float, shuffle_mode: str = "all"):
    if shuffle_mode is None or shuffle_mode == "":
        train_df, valid_df = train_test_split(df, test_size=test_size, shuffle=False)
    elif shuffle_mode == "all":
        train_df, valid_df = train_test_split(df, test_size=test_size, shuffle=True)
    elif shuffle_mode == "in_day":
        assert "dayno" in df.columns, "in_day mode must contains dayno column"
        train_list, test_list = [], []
        dayno_list = sorted(df.dayno.unique())
        for dayno in dayno_list:
            tmp_train, tmp_test = train_test_split(df.query(f"dayno == {dayno}"), test_size=test_size, shuffle=True)
            train_list.append(tmp_train)
            test_list.append(tmp_test)
        train_df, valid_df = pd.concat(train_list), pd.concat(test_list)
    else:
        raise Exception(f"Does not supported shuffle mode = '{shuffle_mode}'")
    return train_df, valid_df


def simple_print_paths(name: str, paths: List[str]):
    if len(paths) > 3:
        s = ',\n    '.join(paths[:2]) + f",\n    ... {len(paths) - 3} more items ..." + '\n    ' + paths[-1]
    else:
        s = ',\n    '.join(paths)
    print(f"Total {name} paths: [\n    {s}\n]")
