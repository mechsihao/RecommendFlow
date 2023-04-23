"""
创建人：MECH
创建时间：2021/11/01
功能描述：根据配置文件构造tfrecord，支持多线程读取，支持从hdfs读取
"""
import glob
import os
import sys
import time

import numpy as np
import tensorflow as tf
from pandas import DataFrame

from config_parser.configuration import Configuration
from config_parser.features import TYPE_STR, TYPE_INT
from utils.util import read_csv
from utils.str_parser import str2list
from utils.hdfs_util import ls_hdfs_paths
import tqdm
from multiprocessing import Process

MAX_THREADS = 64


def _build_int_feature(data):
    """int str会被逗号分成int list"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(i) for i in data.split(",")]))


def _build_float_feature(data):
    """float str会被逗号分成float list"""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(i) for i in data.split(",")]))


def _build_str_feature(data):
    """str会被逗号分成str list
    这里空的字符串都会用 空格 来替换，因为真的空字符串代表padding值，在list不定长时被拿来padding，因此会在下游被mask掉
    """
    data = "" if data == "-1" else str(data)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[i.encode() for i in data.split(",")]))


def _build_bert_feature(data, tokenizer_helper, max_len):
    """一句文本会被存储成两个int list"""
    tok_id, seg_id = tokenizer_helper.encode(str(data), maxlen=max_len)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=tok_id)), tf.train.Feature(int64_list=tf.train.Int64List(value=seg_id))


def _build_embedding_feature(data):
    """
    注意事项:
        - embedding feature 存储 三维以下 的embedding:
        - '|'分割第三维度、';'分割第二维度、','分割第一维度
        - 存储格式为：1,12,3;4,1,2;5,1,3|1,12,3;4,1,2;5,1,3
        - 需自行保证embedding维度的一致，parser不会校验！！！
        - 目前浮点数都是用float32，int使用int64
    """
    if "|" in data:
        value = [[str2list(i, trans_type='float32') for i in str2list(line, sep=";")] for line in str2list(data, sep="|")]
    elif ";" in data:
        value = [str2list(i, trans_type='float32') for i in str2list(data, sep=";")]
    else:
        value = str2list(data, trans_type='float32')
    tensor = tf.convert_to_tensor(np.array(value))
    bytes_str = tf.io.serialize_tensor(tensor).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_str]))


def _build_image_feature(value):
    """输入为一个图像的二进制串"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_or_ignore_row_image_data(row, name, na="-1"):
    """
    目前仅支持将图像存储的到一个特定的目录下，并且将该图片的地址存储到df的一列中，然后使用tf.io.read_file读取该图片，读取后的格式为二进制字符串，然后解析成array的方式
    """
    return tf.io.read_file(row[name]).numpy() if os.path.isfile(row[name]) else b"" if name in row else na


def get_or_ignore_row_data(row, name, na="-1"):
    """忽略缺失的列，生成默认值"""
    return row[name] if name in row else na


def build_tfrecord(row: str, conf: Configuration):
    """
    构建tfrecord存储前的pb序列化对象
    :param row: 某个单独的数据
    :param conf: dict,必须通过load_config来导入
    :return: pb序列化对象
    """
    features = {}
    for feature in conf.features:
        # 这里tfrecord数据里面只存储 train_cols 中的列，cols中的列仅为 记录，不产生实际效果
        if feature.is_numeric():
            features[feature] = _build_float_feature(get_or_ignore_row_data(row, feature))
        elif feature.is_discrete():
            features[feature] = _build_float_feature(get_or_ignore_row_data(row, feature))
        elif feature.is_hashing():
            features[feature] = _build_str_feature(get_or_ignore_row_data(row, feature))
        elif feature.is_lookup() and feature.type == TYPE_INT:
            features[feature] = _build_int_feature(get_or_ignore_row_data(row, feature))
        elif feature.is_lookup() and feature.type == TYPE_STR:
            features[feature] = _build_str_feature(get_or_ignore_row_data(row, feature))
        elif feature.is_embedding():
            features[feature] = _build_embedding_feature(get_or_ignore_row_data(row, feature))
        elif feature.is_image():
            features[feature] = _build_image_feature(get_or_ignore_row_image_data(row, feature))
        elif feature.is_bert_encode():
            features[feature] = _build_str_feature(get_or_ignore_row_data(row, feature))
        elif feature.is_token_id():
            features[feature] = _build_int_feature(get_or_ignore_row_data(row, feature))
        else:
            raise Exception(f"Unsupported deal method feature: {feature}")

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def transform_dataframe(df: DataFrame) -> DataFrame:
    """
    这里涉及到对data frame的预处理环节，目前是不进行处理，在生成tfrecord时就处理好，后续可以对极特殊的情况在这里进行处理
    :param df: 待处理df
    :return: 处理后df
    """
    return df


def generator(src, conf: Configuration):
    df = read_csv(src, sep="\t")
    df = transform_dataframe(df)
    for i, row in tqdm.tqdm(df.iterrows()):
        example = build_tfrecord(row, conf)
        yield example


def dump_tfrecord_data(input_file, out_file, conf):
    print("Source File:", input_file)
    print("Target File:", out_file)
    with tf.io.TFRecordWriter(out_file, "GZIP") as writer:
        for tmp_example in generator(input_file, conf):
            writer.write(tmp_example)


if __name__ == "__main__":
    print("args is: ", sys.argv)
    if len(sys.argv) != 4:
        raise AssertionError(f"Usage: python {sys.argv[0]} conf_path src_file_pattern output_dir")
    else:
        script = sys.argv[0]
        config_path = sys.argv[1]
        src_file_pattern = sys.argv[2]
        output_dir = sys.argv[3]

    feature_conf = Configuration(config_path)

    if src_file_pattern.startswith("hdfs://"):
        print("Read from hdfs: {}".format(src_file_pattern))
        file_paths = sorted(ls_hdfs_paths(src_file_pattern))
    else:
        print("Read from local: {}".format(src_file_pattern))
        file_paths = sorted(glob.glob(src_file_pattern))

    print(f"Total tfrecord paths: {','.join(file_paths)}")
    print(f"Save data to Directory: {output_dir}")

    s0 = time.time()
    if len(file_paths) == 1:
        # 对于一个文件单线程生成tfrecord
        src_file = file_paths[0]
        dst_file_name = src_file.split("/")[-1] + ".0.tfrecord.gz"
        dump_tfrecord_data(src_file, dst_file_name, feature_conf)
    else:
        for batch in range(len(file_paths) // MAX_THREADS + 1):
            s = time.time()
            args_list = []
            for ind, src_file in enumerate(file_paths[MAX_THREADS * batch: MAX_THREADS * (batch + 1)]):
                file_name = src_file.split("/")[-1]
                dst_file_name = file_name + f"{(MAX_THREADS * batch) + ind}.tfrecord.gz"
                dst_file_path = output_dir + "/" + dst_file_name
                args_list.append([src_file, dst_file_path, feature_conf])

            # 多线程加速生成多个tfrecord
            coord = tf.train.Coordinator()
            processes = []
            for thread_index in range(len(args_list)):
                src_file, dst_file_path, feature_conf, bert_tokenizer = args_list[thread_index]
                args = (src_file, dst_file_path, feature_conf, bert_tokenizer)
                p = Process(target=dump_tfrecord_data, args=args)
                p.start()
                processes.append(p)
            coord.join(processes)
            print(f"Batch-{batch}: {len(args_list)} files save to tfrecord success, Cost time: {time.time() - s}")
    print(f"Save to tfrecord finish, Cost total time: {time.time() - s0}")
