import tensorflow as tf
from tensorflow.keras import layers
from config_parser.configuration import Configuration
from backend.layers.preprocess_layers import DoubleHashingEmbedding, BertEncode, LookupEmbedding, DiscreteEmbedding


def get_preprocess_layers(conf: Configuration):
    preprocess_layers = {}
    for feature in conf.train_features:
        if feature.is_hashing():
            hashing = DoubleHashingEmbedding(
                num_bins=feature.vocab_size,
                output_dim=feature.embedding_dim,
                seeds=feature.hash_seeds,
                mask_value="",
                mask_zero=True,
                combiner=feature.pooling.value,
                name=f"hashing_{feature.name}"
            )
            preprocess_layers[feature.name] = hashing
        elif feature.is_lookup():
            lookup_embedding = LookupEmbedding(
                embedding_dim=feature.embedding_dim,
                dtype=feature.type,
                vocabs=feature.vocabs,
                vocab_size=feature.vocab_size,
                pooling=feature.pooling.value,
                name=f"lookup_{feature.name}"
            )
            preprocess_layers[feature.name] = lookup_embedding
        elif feature.is_discrete():
            # 注：分箱特征无法处理缺失值，因为目前所有缺失值被填充为-1，所以分箱特征需要自行检查缺失值的占比，以防影响模型效果
            disc_embedding = DiscreteEmbedding(
                embedding_dim=feature.embedding_dim,
                vocabs=feature.vocabs,
                vocab_size=feature.vocab_size,
                pooling=feature.pooling.value,
                name=f"discrete_{feature.name}"
            )
            preprocess_layers[feature.name] = disc_embedding
        elif feature.is_bert_encode():
            embedding = BertEncode(dict_path=feature.vocabs, name=f"bert_encode_{feature.name}")
            preprocess_layers[feature.name] = embedding
        else:
            pass

    return preprocess_layers


def create_model_inputs(conf: Configuration):
    feature_input_dict = {}
    for feature in conf.train_features:
        input_unit = layers.Input(name=feature, shape=(), dtype=tf.string)
        if feature.is_hashing():
            hashing = DoubleHashingEmbedding(
                num_bins=feature.vocab_size,
                output_dim=feature.embedding_dim,
                seeds=feature.hash_seeds,
                mask_value="",
                mask_zero=True,
                combiner=feature.pooling.value,
                name=f"hashing_{feature.name}"
            )
            feature_input_dict[feature.name] = tf.keras.Sequential([input_unit, hashing], name=f"input_hashing_{feature.name}")
        elif feature.is_lookup():
            lookup_embedding = LookupEmbedding(
                embedding_dim=feature.embedding_dim,
                dtype=feature.type,
                vocabs=feature.vocabs,
                vocab_size=feature.vocab_size,
                pooling=feature.pooling.value,
                name=f"lookup_{feature.name}"
            )
            feature_input_dict[feature.name] = tf.keras.Sequential([input_unit, lookup_embedding], name=f"input_lookup_{feature.name}")
        elif feature.is_discrete():
            disc_embedding = DiscreteEmbedding(
                embedding_dim=feature.embedding_dim,
                vocabs=feature.vocabs,
                vocab_size=feature.vocab_size,
                pooling=feature.pooling.value,
                name=f"discrete_{feature.name}"
            )
            feature_input_dict[feature.name] = tf.keras.Sequential([input_unit, disc_embedding], name=f"input_disc_{feature.name}")
        elif feature.is_bert_encode():
            embedding = BertEncode(dict_path=feature.vocabs, name=f"bert_encode_{feature.name}")
            feature_input_dict[feature.name] = tf.keras.Sequential([input_unit, embedding], name=f"input_disc_{feature.name}")
        else:
            feature_input_dict[feature.name] = input_unit
    return feature_input_dict
