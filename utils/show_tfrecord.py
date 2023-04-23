"""
创建人：李思浩（80302421）
创建时间：2021/11/01
功能描述：根据配置文件构造tfrecord
"""
import sys
from functools import partial
import tensorflow as tf
import numpy as np
from backend.core.dataloader import parse_example, build_feature_description
from config_parser.configuration import Configuration


if __name__ == "__main__":
    print("args is: ", sys.argv)
    if len(sys.argv) != 4 and len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} conf_file tfrecord_file [show_lines]")
        exit(1)

    script = sys.argv[0]
    config_file = sys.argv[1]
    tfrecord_file = sys.argv[2]
    show_lines = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    feature_conf = Configuration(config_file)
    feature_desc = build_feature_description(feature_conf)

    dataset = tf.data.TFRecordDataset(
        [tfrecord_file],
        compression_type="GZIP",
        buffer_size=1024,
        num_parallel_reads=1
    ).repeat(1)

    print("Start Show Binary Data Detail:")
    dataset = dataset.batch(4).map(
        partial(
            parse_example,
            feature_description=feature_desc,
            label_names=feature_conf.features.labels,
            feature_conf=feature_conf
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).prefetch(
        64
    ).take(show_lines)

    for i, (features, labels) in enumerate(dataset):
        for col in features.keys():
            if col in feature_conf.features.hasing_feature_names:
                features[col] = np.array([[d.decode() for d in line] for line in features[col].numpy()])
                print(f"{i}-batch: Feature = {col}, Values = \n{features[col]}")
            elif col in feature_conf.features.lookup_feature_names:
                features[col] = np.array([[d.decode() for d in line] for line in features[col].numpy()])
                print(f"{i}-batch: Feature = {col}, Values = \n{features[col]}")
            else:
                print(f"{i}-batch: Feature = {col}, Values = \n{features[col]}")

        for col in feature_conf.features.label_names:
            print(f"{i}-batch: Label = {col}, Values = \n{labels[col]}")
