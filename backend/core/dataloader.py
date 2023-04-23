"""
创建人：MECH
创建时间：2023/02/01
功能描述：数据读取组件，目前仅支持读取tfrecord和结构化的csv
"""

import glob
import random
import difflib
from typing import List
from functools import partial

import tensorflow as tf
import tensorflow_io as tfio  # 一定需要这样调用，否则会报错 File system scheme 'hdfs' not implemented，也要注意自动reformat有可能将其删除，一定注意

from config_parser.config_proto import FeatureDeal
from config_parser.configuration import Configuration
from config_parser.features import Feature
from utils.hdfs_util import ls_hdfs_paths
from utils.util import simple_print_paths


def build_feature_description(conf: Configuration):
    """
    按照配置文件解析tfrecord的输入，这里相当于Keras的Input
    :param conf: 配置文件Dict 类型
    """
    feature_desc = {}
    for f in conf.train_features:
        if f.deal == FeatureDeal.Numeric:
            feature_desc[f.name] = tf.io.FixedLenFeature(shape=(), dtype=f.type, default_value=f.default)
        elif f.deal in (FeatureDeal.Discrete, FeatureDeal.Hashing, FeatureDeal.Lookup):
            feature_desc[f.name] = tf.io.FixedLenSequenceFeature(shape=(), dtype=f.type, allow_missing=True, default_value=f.default)
        elif f.deal in (FeatureDeal.Image, FeatureDeal.Embedding):
            feature_desc[f.name] = tf.io.FixedLenFeature(shape=(), dtype=f.type, default_value=f.default)
        elif f.deal == FeatureDeal.TokenId:
            feature_desc[f.name] = tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.int64, allow_missing=True, default_value=0)
        elif f.deal == FeatureDeal.BertEncode:
            feature_desc[f.name] = tf.io.FixedLenFeature(shape=(1,), dtype=tf.string, default_value="")
        elif f.deal == FeatureDeal.Null:
            feature_desc[f.name] = tf.io.FixedLenFeature(shape=(), dtype=f.type, default_value=f.default)
        else:
            raise Exception(f"Unregister Feature: {f.name}")
    return feature_desc


def get_label_dict(example, label_names: List[Feature]):
    """
    获取label
    :param example:
    :param label_names:
    :return: 返回解析好的label数据
    """
    label_dict = {}
    for col in label_names:
        label_dict[col.name] = example[col.name]
    return label_dict


def second_parser(example, feature_conf: Configuration):
    """
    解析需要二次解析的列，目前支持embedding和图像的解析
    :param example:
    :param feature_conf:
    :return: 返回二次解析好的数据
    """
    for f in feature_conf.features.get_embedding_features():
        example[f.name] = tf.io.parse_tensor(example[f.name], out_type=tf.float32)
    for f in feature_conf.features.get_image_features():
        try:
            example[f.name] = tf.io.decode_jpeg(example[f.name])
        except ValueError:
            example[f.name] = tfio.image.decode_webp(example[f.name])
    return example


def parse_example(example_proto, feature_description: dict, label_names: List[Feature], feature_conf: Configuration = None):
    """
    解析函数
    :param example_proto:
    :param feature_description:
    :param label_names:
    :param feature_conf: 需要二次解析的列，会在second_parse中进行二次解析
    :return: 返回解析好的数据
    """
    example = tf.io.parse_example(example_proto, feature_description)
    labels = get_label_dict(example, label_names)
    example = second_parser(example, feature_conf) if feature_conf.need_parse_second else example
    return example, labels


def parse_patten(path_patten: str, from_hdfs: bool = False, show_simple_info: bool = True):
    """
    读取路径通配模式
    :param path_patten:
    :param from_hdfs:
    :param show_simple_info:
    :return:
    """
    if from_hdfs or path_patten.startswith("hdfs://"):
        print("Read from hdfs: {}".format(path_patten))
        paths = sorted(ls_hdfs_paths(path_patten))
    else:
        print("Read from local: {}".format(path_patten))
        paths = sorted(glob.glob(path_patten))

    if show_simple_info and len(paths) > 5:
        simple_print_paths("tfrecord", paths)
    else:
        s = ',\n    '.join(paths)
        print(f"Total tfrecord paths: [\n    {s}\n]")

    print(f"Total tfrecord nums = {len(paths)}")
    return paths


def shuffle_data(paths, sample_ratio, shuffle):
    """
    文件粒度打乱样本，目前只支持文件粒度
    :param paths:
    :param sample_ratio:
    :param shuffle:
    :return:
    """
    if sample_ratio is not None and sample_ratio != 1.0:
        k = max(int(len(paths) * sample_ratio), 1)
        paths = random.sample(paths, k)
        simple_print_paths(f"Sample {k} tfrecord", paths)

    if shuffle:
        print("Shuffle paths...")
        random.shuffle(paths)
    return paths


def split_data(paths, valid_size, sample_ratio, shuffle):
    """切分训练集和测试集，只支持文件粒度的切分，如果测试集比例不够，则由训练集的部分来补充
    """
    assert valid_size < 1
    train_pool = shuffle_data(paths, sample_ratio, shuffle)
    valid_pool = [path for path in paths if path not in train_pool]

    train_size = max(int(len(train_pool) * (1 - valid_size)), 1)
    valid_size = max(int(len(train_pool) * valid_size), 1)

    train_paths = (train_pool + valid_pool)[:train_size]
    valid_paths = (train_pool + valid_pool)[-valid_size:]

    print(f"All sample {len(paths)} tfrecords")
    simple_print_paths(f"train sample {len(train_paths)} tfrecords", train_paths)
    simple_print_paths(f"valid sample {len(valid_paths)} tfrecords", valid_paths)
    return train_paths, valid_paths


def load_csv(
        csv_path_pattern: str,
        batch_size: int = 64,
        header: bool = True,
        sep: str = '\t',
        use_cols: List[str] = None,
        names: List[str] = None,
        na_value: str = '',
        shuffle: bool = True,
        num_parallel_reads: int = None,
        num_rows_for_inference: int = 100,
        label_name: str = "label",
        ignore_errors: bool = False):
    """
    将csv文件导入成tf dataset类型
    :param csv_path_pattern:
    :param batch_size:
    :param header:
    :param sep:
    :param use_cols:
    :param names:
    :param na_value:
    :param shuffle:
    :param num_parallel_reads:
    :param num_rows_for_inference:
    :param label_name:
    :param ignore_errors:
    :return:
    """
    return tf.data.experimental.make_csv_dataset(
        csv_path_pattern,
        batch_size=batch_size,
        header=header,
        field_delim=sep,
        select_columns=use_cols,
        column_names=names,
        na_value=na_value,
        shuffle=shuffle,
        num_parallel_reads=num_parallel_reads,
        num_rows_for_inference=num_rows_for_inference,
        label_name=label_name,
        ignore_errors=ignore_errors
    )


def load_tfrecord(path_patten,
                  feature_conf,
                  label_names,
                  batch_size,
                  thread_num,
                  sample_ratio=None,
                  compression_type="GZIP",
                  prefetch_buffer_size=1024 * 10,
                  buffer_size=100 * 1024 * 1024,
                  from_hdfs=None,
                  shuffle=True,
                  drop_remainder=False,
                  is_debug=False):
    """
    解析dataset
    :param path_patten: 样本路径，支持通配
    :param feature_conf: 特征描述
    :param label_names: label 名称
    :param batch_size: batch size
    :param thread_num: 读取时的线程数
    :param sample_ratio: 采样率，如果为None则不采样
    :param compression_type: tfrecord文件压缩类型，默认使用GZIP
    :param prefetch_buffer_size: 提前缓存的样本batch数
    :param buffer_size: 缓存空间大小
    :param from_hdfs: 是否从hdfs读取。None表示自动从路径判断
    :param shuffle: 是否打乱文件，只进行文件粒度的打乱
    :param drop_remainder: 是否丢弃最后一个残缺batch
    :param is_debug: 是否是调试模式，调试模式下只会取前50个batch的数据
    """
    paths = parse_patten(path_patten, from_hdfs)

    paths = shuffle_data(paths, sample_ratio, shuffle)

    feature_description = build_feature_description(feature_conf)

    dataset = _get_tfrecord_dataset(paths,
                                    feature_description,
                                    label_names,
                                    batch_size,
                                    thread_num,
                                    compression_type=compression_type,
                                    prefetch_buffer_size=prefetch_buffer_size,
                                    buffer_size=buffer_size,
                                    drop_remainder=drop_remainder,
                                    feature_conf=feature_conf,
                                    is_debug=is_debug)
    return dataset


def load_multi_tfrecord_datasets(path_patten,
                                 feature_conf,
                                 label_names,
                                 batch_size,
                                 thread_num,
                                 file_num_per_dataset=10,
                                 sample_ratio=None,
                                 compression_type="GZIP",
                                 prefetch_buffer_size=1024 * 100,
                                 buffer_size=100 * 1024 * 1024,
                                 from_hdfs=None,
                                 shuffle=True,
                                 drop_remainder=False):
    """
    按照解析file_num_per_dataset，将输入文件划分成若干个datasets，返回generator。适用于单次返回结果太大，需要分批次处理
    :param path_patten: 样本路径，支持通配
    :param feature_conf: 特征配置文件
    :param label_names: label 名称
    :param batch_size: batch size
    :param thread_num: 读取时的线程数
    :param file_num_per_dataset: 每个dataset取的文件part数量
    :param sample_ratio: 采样率，如果为None则不采样
    :param compression_type: tfrecord文件压缩类型，默认使用GZIP
    :param prefetch_buffer_size: 提前缓存的样本batch数
    :param buffer_size: 缓存空间大小
    :param from_hdfs: 是否从hdfs读取。None表示自动从路径判断
    :param shuffle: 是否打乱文件，只进行文件粒度的打乱
    :param drop_remainder: 是否丢弃最后一个残缺batch
    """
    paths = parse_patten(path_patten, from_hdfs)

    paths = shuffle_data(paths, sample_ratio, shuffle)

    feature_description = build_feature_description(feature_conf)

    paths_list = [
        paths[i:i + file_num_per_dataset]
        for i in range(0, len(paths), file_num_per_dataset)
    ]
    for paths in paths_list:
        dataset = _get_tfrecord_dataset(
            paths,
            feature_description,
            label_names,
            batch_size,
            thread_num,
            compression_type=compression_type,
            prefetch_buffer_size=prefetch_buffer_size,
            buffer_size=buffer_size,
            drop_remainder=drop_remainder,
            feature_conf=feature_conf)
        yield dataset


def load_multi_tfrecord_dayno_patten(path_patten,
                                     dayno_patten,
                                     feature_conf,
                                     label_names,
                                     batch_size,
                                     thread_num,
                                     file_num_per_dataset=10,
                                     sample_ratio=None,
                                     compression_type="GZIP",
                                     prefetch_buffer_size=1024 * 100,
                                     buffer_size=100 * 1024 * 1024,
                                     from_hdfs=None,
                                     shuffle=True,
                                     drop_remainder=False,
                                     is_debug=False):
    """
    按照解析file_num_per_dataset，将输入文件划分成若干个datasets，返回generator。适用于单次返回结果太大，需要分批次处理
    :param path_patten: 样本通配路径
    :param dayno_patten: 通配日期
    :param feature_conf: 特征配置文件
    :param label_names: label 名称
    :param batch_size: batch size
    :param thread_num: 读取时的线程数
    :param file_num_per_dataset: 每个dataset取的文件part数量
    :param sample_ratio: 采样率，如果为None则不采样
    :param compression_type: tfrecord文件压缩类型，默认使用GZIP
    :param prefetch_buffer_size: 提前缓存的样本batch数
    :param buffer_size: 缓存空间大小
    :param from_hdfs: 是否从hdfs读取。None表示自动从路径判断
    :param shuffle: 是否打乱文件，只进行文件粒度的打乱
    :param drop_remainder: 是否丢弃最后一个残缺batch
    :param is_debug:
    """
    path_patten = path_patten.replace("YYYYMMMDD", dayno_patten)
    paths = parse_patten(path_patten, from_hdfs)

    paths = shuffle_data(paths, sample_ratio, shuffle)

    feature_description = build_feature_description(feature_conf)

    paths_list = [
        paths[i:i + file_num_per_dataset]
        for i in range(0, len(paths), file_num_per_dataset)
    ]
    for paths in paths_list:
        dataset = _get_tfrecord_dataset(
            paths,
            feature_description,
            label_names,
            batch_size,
            thread_num,
            compression_type=compression_type,
            prefetch_buffer_size=prefetch_buffer_size,
            buffer_size=buffer_size,
            drop_remainder=drop_remainder,
            feature_conf=feature_conf,
            is_debug=is_debug
        )
        yield dataset


def load_tfrecord_dayno_patten(path_patten,
                               dayno_patten,
                               feature_conf,
                               label_names,
                               batch_size,
                               thread_num,
                               sample_ratio=None,
                               compression_type="GZIP",
                               prefetch_buffer_size=1024 * 10,
                               buffer_size=100 * 1024 * 1024,
                               from_hdfs=None,
                               shuffle=True,
                               drop_remainder=False,
                               is_debug=False):
    """
    解析dataset
    :param path_patten: 样本路径，必须包含YYYYMMDD代表日期的位置，支持通配
    :param dayno_patten: 日期通配符
    :param feature_conf: 特征描述
    :param label_names: label 名称
    :param batch_size: batch size
    :param thread_num: 读取时的线程数
    :param sample_ratio: 采样率，如果为None则不采样
    :param compression_type: tfrecord文件压缩类型，默认使用GZIP
    :param prefetch_buffer_size: 提前缓存的样本batch数
    :param buffer_size: 缓存空间大小
    :param from_hdfs: 是否从hdfs读取。None表示自动从路径判断
    :param shuffle: 是否打乱文件，只进行文件粒度的打乱
    :param drop_remainder: 是否丢弃最后一个残缺batch
    :param is_debug: 是否是调试模式，调试模式下只会取前50个batch的数据
    """
    assert "YYYYMMDD" in path_patten, "You must use 'YYYYMMDD' to occupied dayno place"
    path_patten = path_patten.replace("YYYYMMDD", dayno_patten)
    train_paths = parse_patten(path_patten, from_hdfs)

    paths = shuffle_data(train_paths, sample_ratio, shuffle)

    feature_description = build_feature_description(feature_conf)
    train_dataset = _get_tfrecord_dataset(paths,
                                          feature_description,
                                          label_names,
                                          batch_size,
                                          thread_num,
                                          compression_type=compression_type,
                                          prefetch_buffer_size=prefetch_buffer_size,
                                          buffer_size=buffer_size,
                                          drop_remainder=drop_remainder,
                                          feature_conf=feature_conf,
                                          is_debug=is_debug)
    return train_dataset


def load_tfrecord_dayno_patten_split(path_patten,
                                     dayno_patten,
                                     feature_conf: Configuration,
                                     label_names,
                                     batch_size,
                                     thread_num,
                                     sample_ratio=None,
                                     compression_type="GZIP",
                                     prefetch_buffer_size=1024 * 10,
                                     buffer_size=100 * 1024 * 1024,
                                     from_hdfs=None,
                                     shuffle=True,
                                     valid_size=0.1,
                                     drop_remainder=False,
                                     is_debug=False):
    """
    解析dataset
    :param path_patten: 样本路径，必须包含YYYYMMDD代表日期的位置，支持通配
    :param dayno_patten: 日期通配符
    :param feature_conf: 特征描述
    :param label_names: label 名称
    :param batch_size: batch size
    :param thread_num: 读取时的线程数
    :param sample_ratio: 采样率，如果为None则不采样
    :param compression_type: tfrecord文件压缩类型，默认使用GZIP
    :param prefetch_buffer_size: 提前缓存的样本batch数
    :param buffer_size: 缓存空间大小
    :param from_hdfs: 是否从hdfs读取。None表示自动从路径判断
    :param shuffle: 是否打乱文件，只进行文件粒度的打乱
    :param drop_remainder: 是否丢弃最后一个残缺batch
    :param valid_size: 验证集大小, 如果文件数*valid_size小于1则从训练集中采样，不过这样会有重叠
    :param is_debug: 是否是调试模式，调试模式下只会取前50个batch的数据
    """
    assert "YYYYMMDD" in path_patten, "You must use 'YYYYMMDD' to occupied dayno place"
    path_patten = path_patten.replace("YYYYMMDD", dayno_patten)
    paths = parse_patten(path_patten, from_hdfs)

    train_paths, valid_paths = split_data(paths, valid_size, sample_ratio, shuffle)

    feature_description = build_feature_description(feature_conf)
    train_dataset = _get_tfrecord_dataset(train_paths,
                                          feature_description,
                                          label_names,
                                          batch_size,
                                          thread_num,
                                          compression_type=compression_type,
                                          prefetch_buffer_size=prefetch_buffer_size,
                                          buffer_size=buffer_size,
                                          drop_remainder=drop_remainder,
                                          feature_conf=feature_conf,
                                          is_debug=is_debug)

    valid_dataset = _get_tfrecord_dataset(valid_paths,
                                          feature_description,
                                          label_names,
                                          batch_size,
                                          thread_num,
                                          compression_type=compression_type,
                                          prefetch_buffer_size=prefetch_buffer_size,
                                          buffer_size=buffer_size,
                                          drop_remainder=drop_remainder,
                                          feature_conf=feature_conf,
                                          is_debug=is_debug)
    return train_dataset, valid_dataset


def load_multi_tfrecord_datasets_dict(path_patten,
                                      feature_conf,
                                      label_names,
                                      batch_size,
                                      thread_num,
                                      sample_ratio=None,
                                      compression_type="GZIP",
                                      prefetch_buffer_size=1024 * 100,
                                      buffer_size=100 * 1024 * 1024,
                                      from_hdfs=None,
                                      shuffle=True,
                                      drop_remainder=False,
                                      is_debug=False):
    """
    按照解析file_num_per_dataset，将输入文件划分成若干个datasets，返回generator。适用于单次返回结果太大，需要分批次处理
    :param path_patten: 样本路径，支持通配
    :param feature_conf: 特征描述
    :param label_names: label 名称
    :param batch_size: batch size
    :param thread_num: 读取时的线程数
    :param sample_ratio: 采样率，如果为None则不采样
    :param compression_type: tfrecord文件压缩类型，默认使用GZIP
    :param prefetch_buffer_size: 提前缓存的样本batch数
    :param buffer_size: 缓存空间大小
    :param from_hdfs: 是否从hdfs读取。None表示自动从路径判断
    :param shuffle: 是否打乱文件，只进行文件粒度的打乱
    :param drop_remainder: 是否丢弃最后一个残缺batch
    :param is_debug:
    :return Dict[pattern_key, tfrecord]
    """
    paths = parse_patten(path_patten, from_hdfs)

    paths = shuffle_data(paths, sample_ratio, shuffle)

    d = difflib.Differ()

    feature_description = build_feature_description(feature_conf)

    res = {}
    for path in paths:
        diff = d.compare(path_patten, path)
        match_patten = "".join(list([i.split("+")[1].strip() for i in diff if "+" in i]))
        dataset = _get_tfrecord_dataset(
            path,
            feature_description,
            label_names,
            batch_size,
            thread_num,
            compression_type=compression_type,
            prefetch_buffer_size=prefetch_buffer_size,
            buffer_size=buffer_size,
            drop_remainder=drop_remainder,
            feature_conf=feature_conf,
            is_debug=is_debug)
        res[match_patten] = dataset
    print("Match pattens: {}".format(", ".join(list(res.keys()))))
    return res


def _get_tfrecord_dataset(paths,
                          feature_description,
                          label_names,
                          batch_size,
                          thread_num,
                          compression_type="GZIP",
                          prefetch_buffer_size=1024 * 100,
                          buffer_size=100 * 1024 * 1024,
                          drop_remainder=False,
                          feature_conf=None,
                          is_debug=False
                          ):
    """
    解析dataset功能函数
    :param paths: 样本路径列表
    :param feature_description: 特征描述
    :param label_names: label 名称
    :param batch_size: batch size
    :param thread_num: 读取时的线程数
    :param compression_type: tfrecord文件压缩类型，默认使用GZIP
    :param prefetch_buffer_size: 提前缓存的样本batch数
    :param buffer_size: 缓存空间大小
    :param feature_conf: 配置文件
    :param is_debug: 是否调试模式
    """
    assert len(paths) > 0, "Paths must not be empty"
    dataset = tf.data.TFRecordDataset(paths,
                                      compression_type=compression_type,
                                      buffer_size=buffer_size,
                                      num_parallel_reads=thread_num).repeat(1)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder).map(
        partial(parse_example,
                feature_description=feature_description,
                label_names=label_names,
                feature_conf=feature_conf
                ),
        num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(prefetch_buffer_size) if not is_debug else dataset.take(10)
