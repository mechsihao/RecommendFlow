from typing import Union

import pandas as pd
import tensorflow as tf
from bert4keras.models import build_transformer_model
from tensorflow import keras
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Lambda


def get_bert(conf_path, ckpt_path, is_trainable=True, name=None):
    bert = build_transformer_model(conf_path, ckpt_path, name=name)
    for layer in bert.layers:
        layer.trainable = is_trainable
    return bert


def get_bert_models_dict(bert_tower_conf):
    bert_model_dict = {}
    for tower_name, tower_conf in bert_tower_conf.items():
        input_features, pooling = tower_conf['cols'],  tower_conf['pooling']
        conf_path, ckpt_path = tower_conf['conf_path'], tower_conf['ckpt_path']

        bert = get_bert_model(conf_path, ckpt_path, pooling, bert_tower_conf[tower_name]["finetune"])

        for input_feature in input_features:
            assert input_feature not in bert_model_dict, f"Encode input = {input_feature}, do not support overlap in " \
                                                         f"bert_model_dict = {bert_model_dict.keys()}"
            output = keras.layers.Lambda(lambda x: x[:, 0])(bert.output)
            bert_model = keras.models.Model(bert.inputs, output, name=f"bert_encoder_{pooling}_{tower_name}_{input_feature}")
            bert_model_dict[input_feature] = bert_model

    return bert_model_dict


def get_siamese_bert_model(query_features, doc_features, conf_path, ckpt_path, is_trainable=True, name=None):
    # bert共享
    bert_model = build_transformer_model(conf_path, ckpt_path, maxlen=None)
    for layer in bert_model.layers:
        layer.trainable = is_trainable

    # query塔
    input_list = []
    output_list = []
    for query_feature in query_features:
        token_input = Input(shape=(None, ), name=f"{name}_{query_feature}_tok_input")
        segment_input = Input(shape=(None, ), name=f"{name}_{query_feature}_seg_input")
        x = bert_model([token_input, segment_input])
        out = Lambda(lambda t: t[:, 0], name=f"{name}_{query_feature}_bert_output")(x)
        input_list.append([token_input, segment_input])
        output_list.append(out)
    query_bert = tf.keras.models.Model(input_list, output_list, name=name)

    # doc塔
    input_list = []
    output_list = []
    for doc_feature in doc_features:
        token_input = Input(shape=(None,), name=f"{name}_{doc_feature}_tok_input")
        segment_input = Input(shape=(None,), name=f"{name}_{doc_feature}_seg_input")
        x = bert_model([token_input, segment_input])
        out = Lambda(lambda t: t[:, 0], name=f"{name}_{doc_feature}_bert_output")(x)
        input_list.append([token_input, segment_input])
        output_list.append(out)
    doc_bert = tf.keras.models.Model(input_list, output_list, name=name)
    return query_bert, doc_bert


def get_bert_model(conf_path, ckpt_path, maxlen, is_trainable, out_layer: int = -1):
    """
    可以指定第几层输出的bert
    :param conf_path:
    :param ckpt_path:
    :param maxlen:
    :param is_trainable:
    :param out_layer:
    :return:
    """
    base = build_transformer_model(conf_path, ckpt_path, maxlen=maxlen if maxlen not in (None, False, "None") else None)
    for layer in base.layers:
        layer.trainable = is_trainable

    bert_df = pd.DataFrame(base.get_config()["layers"])
    out_layer_name_list = bert_df[bert_df.name.apply(lambda x: x.endswith("FeedForward-Norm"))].name.tolist()
    if -len(out_layer_name_list) < out_layer < len(out_layer_name_list):
        out_layer_name = out_layer_name_list[out_layer]
    else:
        raise ValueError(f"Out layer must be a int less than {len(out_layer_name_list)}")
    out_layer = base.get_layer(out_layer_name)
    return keras.models.Model(base.inputs, out_layer.output)


def get_multi_input_bert_model(feature_list,
                               conf_path,
                               ckpt_path,
                               is_trainable=True,
                               pooling: Union[str, int, None] = "cls",
                               out_layer: int = -1,
                               maxlen=None,
                               name=None):
    """
    多输入多输出共享的bert权重
    :param feature_list:
    :param conf_path:
    :param ckpt_path:
    :param is_trainable:
    :param pooling:
    :param out_layer:
    :param maxlen:
    :param name:
    :return:
    """
    bert_model = get_bert_model(conf_path, ckpt_path, maxlen, is_trainable, out_layer)
    input_list, output_list = [], []
    for feature in feature_list:
        token_input = Input(shape=(None, ), name=f"{feature}_tok")
        segment_input = Input(shape=(None, ), name=f"{feature}_seg")
        input_list.append([token_input, segment_input])
        x = bert_model([token_input, segment_input])
        if pooling is None:
            out = Lambda(lambda t: t, name=f"{feature}_all_out")(x)
        elif isinstance(pooling, str) and pooling.lower() == "cls":
            out = Lambda(lambda t: t[:, 0], name=f"{feature}_cls_out")(x)
        elif isinstance(pooling, str) and pooling.lower() == "avg":
            out = keras.layers.GlobalAveragePooling1D(name=f"{feature}_avg_out")(x)
        elif isinstance(pooling, str) and pooling.lower() == "max":
            out = keras.layers.GlobalMaxPooling1D(name=f"{feature}_max_out")(x)
        elif isinstance(pooling, str) and pooling.lower() == "sum":
            out = keras.layers.GlobalSumPooling1D(name=f"{feature}_sum_out")(x)
        elif isinstance(pooling, int):
            out = Lambda(lambda t: t[:, int(pooling)], name=f"{feature}_pos_{pooling}_out")(x)
        else:
            raise ValueError("")
        output_list.append(out)

    return tf.keras.models.Model(input_list, output_list, name=name)
