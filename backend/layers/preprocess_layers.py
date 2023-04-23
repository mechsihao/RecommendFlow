"""
创建人：MECH
创建时间：2021/11/19
功能描述：将keras底层的Layer继承后，加工处理成更方便框架使用的Layer
"""
from typing import List

import tensorflow as tf
from bert4keras.tokenizers import Tokenizer
from tensorflow.python import keras
from tensorflow.python.keras.layers import Embedding, Lambda, Hashing, StringLookup, IntegerLookup, Discretization

from config_parser.features import TYPE_STR, TYPE_INT


class EmbeddingBag(keras.layers.Layer):
    """EmbeddingBag 的实现
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 mask_zero=False,
                 combiner: str = "sum",
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 **kwargs):
        super(EmbeddingBag, self).__init__(**kwargs)
        self.combiner = combiner
        self.embedding_layer = Embedding(input_dim,
                                         output_dim,
                                         embeddings_initializer=embeddings_initializer,
                                         embeddings_regularizer=embeddings_regularizer,
                                         activity_regularizer=activity_regularizer,
                                         embeddings_constraint=embeddings_constraint,
                                         mask_zero=mask_zero,
                                         input_length=combiner
                                         )
        self.combiner_layer = Lambda(lambda t: self.get_combiner(t))
        self.support_pooling = ["null", "sum", "min", "max", "avg", "first", "last"]

    @tf.function
    def get_combiner(self, x):
        if self.combiner == "null":
            # 注意：
            #   - 如果combiner也就是pooling方式如果为null，则会全向量返回，如果不定长，则会用0向量来填充，请自行保证特征数据定长
            #   - 且返回值为一个[batch, seq_len, emb_dim]的三维tensor，请注意如果将这个特征不佳特殊处理无法与其他二维tensor一起concat
            out = Lambda(lambda t: t)(x)
        elif self.combiner == "first":
            out = Lambda(lambda t: t[0])(x)
        elif self.combiner == "last":
            out = Lambda(lambda t: t[-1])(x)
        elif self.combiner == "sum":
            out = tf.reduce_sum(x, axis=1)
        elif self.combiner == "min":
            out = tf.reduce_min(x, axis=1)
        elif self.combiner == "max":
            out = tf.reduce_max(x, axis=1)
        elif self.combiner == "avg":
            out = tf.reduce_mean(x, axis=1)
        else:
            raise ValueError(f"Do not support combiner = '{self.combiner}', supported: [{', '.join(self.support_pooling)}]")
        return out

    def call(self, inputs, *args, **kwargs):
        embedding_matrix = self.embedding_layer(inputs)
        return self.combiner_layer(embedding_matrix)

    def __call__(self, inputs, *args, **kwargs):
        return self.call(inputs, *args, **kwargs)

    def get_config(self):
        config = super(EmbeddingBag, self).get_config()
        config.update({"combiner": self.combiner})
        return config


class DoubleHashingEmbedding(keras.layers.Layer):
    """二阶Hash实现 + EmbeddingBag
    """
    def __init__(self, num_bins, output_dim, seeds, combiner, mask_value=None, mask_zero=False, name=""):
        if num_bins is None or num_bins <= 0:
            raise ValueError('`num_bins` cannot be `None` or non-positive values.')
        super(DoubleHashingEmbedding, self).__init__(name=name)
        self.num_bins = num_bins
        self.mask_value = mask_value
        self.seeds = [seeds, seeds + 7] if isinstance(seeds, int) else seeds  # 7是个随机加的数字，可以修改
        self.hash1 = Hashing(num_bins, mask_value=mask_value, name=f"{name}_hashing1", salt=seeds[0])
        self.hash2 = Hashing(num_bins, mask_value=mask_value, name=f"{name}_hashing2", salt=seeds[1])
        self.emb1 = EmbeddingBag(num_bins, output_dim, mask_zero, combiner=combiner, name=f"{name}_embedding_bag1")
        self.emb2 = EmbeddingBag(num_bins, output_dim, mask_zero, combiner=combiner, name=f"{name}_embedding_bag2")

    def call(self, inputs, *args, **kwargs):
        h1, h2 = self.hash1(inputs), self.hash2(inputs)
        embedding1, embedding2 = self.emb1(h1), self.emb2(h2)
        return tf.concat([embedding1, embedding2], axis=1, name=self.name + "_concat")

    def __call__(self, inputs, *args, **kwargs):
        return self.call(inputs, *args, **kwargs)

    def get_config(self):
        config = super(DoubleHashingEmbedding, self).get_config()
        config.update({"combiner": self.combiner})
        config.update({"seeds": self.seeds})
        return config


class BertEncode(keras.layers.Layer):
    """BertTokenizer，实现自苏剑林版bert4keras
    """
    def __init__(self, dict_path: str,  max_len: int = None, name: str = None):
        super(BertEncode, self).__init__(name=name)
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)
        self.max_len = max_len

    def call(self, inputs, training=False, **kwargs):
        batch_token_ids, batch_segment_ids = [], []
        for text in tf.squeeze(inputs).numpy():
            # 这里一定需要保证输入的为str_list，并且list中只有一个元素
            token_ids, segment_ids = self.tokenizer.encode(text.decode("utf-8"), maxlen=self.max_len)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
        return tf.ragged.constant(batch_token_ids).to_tensor(), tf.ragged.constant(batch_segment_ids).to_tensor()

    def get_config(self):
        config = super(BertEncode, self).get_config()
        config.update({"max_len": self.max_len})
        return config

    def __call__(self, inputs, training=False, **kwargs):
        return self.call(inputs, training, **kwargs)


class LookupEmbedding(keras.layers.Layer):
    def __init__(self,
                 embedding_dim: int,
                 dtype: str,
                 vocabs: List[int],
                 vocab_size: int = None,
                 pooling: str = "sum",
                 name: str = None):
        super(LookupEmbedding, self).__init__(name=name)
        self.vocabulary = vocabs
        self.pooling = pooling
        vocab_size = vocab_size or len(vocabs) + 1

        if dtype == TYPE_STR:
            self.lookup_id = StringLookup(vocabulary=vocabs, output_mode="int", name=name + "_lookup")
        elif dtype == TYPE_INT:
            self.lookup_id = IntegerLookup(vocabulary=vocabs, output_mode="int", name=name + "_lookup")
        else:
            raise ValueError(f"Unsupported type for lookup feature: {dtype}")

        self.embedding = EmbeddingBag(vocab_size, embedding_dim, True, combiner=pooling, name=name + "_embedding")

    def call(self, inputs, *args, **kwargs):
        self.update_lookup_layer(inputs)
        lookup_id = self.lookup_id(inputs)
        return self.embedding(lookup_id)

    def get_vocabulary(self):
        return self.lookup_id.get_vocabulary()

    def get_config(self):
        config = super(LookupEmbedding, self).get_config()
        config.update({"vocabulary": self.vocabulary})
        config.update({"pooling": self.pooling})
        return config


class DiscreteEmbedding(keras.layers.Layer):
    """
    注：分箱特征无法处理缺失值，因为目前所有缺失值被填充为-1，所以分箱特征需要自行检查缺失值的占比，以防影响模型效果
    """
    def __init__(self,
                 embedding_dim: int,
                 vocabs: List[int],
                 vocab_size: int = None,
                 pooling: str = "sum",
                 name: str = None):
        super(DiscreteEmbedding, self).__init__(name=name)
        self.vocabulary = vocabs
        vocab_size = vocab_size or len(vocabs) + 1
        self.pooling = pooling
        self.nums_id = Discretization(bin_boundaries=vocabs)
        self.embedding = EmbeddingBag(vocab_size, embedding_dim, True, combiner=pooling, name=name + "_disc_lookup_embedding")

    def call(self, inputs, *args, **kwargs):
        nums_id = self.nums_id(inputs)
        return self.embedding(nums_id)

    def get_vocabulary(self):
        return self.nums_id.bin_boundaries

    def get_config(self):
        config = super(DiscreteEmbedding, self).get_config()
        config.update({"vocabulary": self.vocabulary})
        config.update({"pooling": self.pooling})
        return config
