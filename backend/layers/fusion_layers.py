import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import initializers, regularizers, constraints, backend as K


class AttentionFusion(keras.layers.Layer):
    """
    Attention Fusion的实现，推理和训练状态有一点不同，推理状态下会记录平均权重，查看效果
    论文：Que2Search
    """

    def __init__(self,
                 input_dim: int,
                 channel_num: int,
                 initializer='glorot_uniform',
                 regularizer=None,
                 constraint=None,
                 is_norm: bool = True,
                 name: str = None):
        super(AttentionFusion, self).__init__(name=name)
        self.input_dim = input_dim
        self.channel_num = channel_num
        self.is_norm = is_norm
        self.W = self.add_weight(
            name="W",
            shape=[input_dim * channel_num, channel_num],
            initializer=initializers.get(initializer),
            regularizer=regularizers.get(regularizer),
            constraint=constraints.get(constraint),
            trainable=True
        )
        self.attention = None
        self.infer_weights = self.add_weight("infer_weights", [1, channel_num], initializer=initializers.get("zeros"), trainable=False)

    def call(self, inputs, training=False, **kwargs):
        embedding = K.concatenate(inputs)
        assert len(embedding.shape) == 2 and embedding.shape[
            1] == self.input_dim * self.channel_num, f"get shape(?, {embedding.shape[-1]}), expect(?, {self.input_dim * self.channel_num})"

        self.attention = tf.nn.softmax(tf.matmul(embedding, self.W))
        out_matrix = tf.stack(inputs) * tf.expand_dims(tf.transpose(self.attention), -1)
        out = tf.reduce_sum(out_matrix, axis=0)

        self.infer_weights.assign_add(tf.reduce_sum(self.attention, axis=0, keepdims=True))

        return tf.nn.l2_normalize(out, axis=1) if self.is_norm else out

    def init_attention(self):
        # 将infer_weights清零， 每一次推理之间可以使用一次清零来整体计算权重占比
        self.infer_weights.assign_sub(self.infer_weights.numpy())

    def get_fusion_weights(self):
        return self.infer_weights.numpy() / self.infer_weights.numpy().sum(axis=1)

    def get_config(self):
        config = super(AttentionFusion, self).get_config()
        config.update({"channel_num": self.channel_num})
        return config

    def __call__(self, inputs, training=False, **kwargs):
        return self.call(inputs, training, **kwargs)
