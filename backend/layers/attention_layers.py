import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import Layer, Dense, Dropout

from layer_utils import split_heads, scaled_dot_product_attention


class SoftAttention(object):
    """
    Layer to compute local inference between two encoded sentences a and b.
    """

    def __call__(self, inputs):
        a = inputs[0]
        b = inputs[1]

        attention = keras.layers.Lambda(self._attention,
                                        output_shape=self._attention_output_shape,
                                        arguments=None)(inputs)

        align_a = keras.layers.Lambda(self._soft_alignment,
                                      output_shape=self._soft_alignment_output_shape,
                                      arguments=None)([attention, a])
        align_b = keras.layers.Lambda(self._soft_alignment,
                                      output_shape=self._soft_alignment_output_shape,
                                      arguments=None)([attention, b])

        return align_a, align_b

    @staticmethod
    def _attention(inputs):
        """
        Compute the attention between elements of two sentences with the dot
        product.
        Args:
            inputs: A list containing two elements, one for the first sentence
                    and one for the second, both encoded by a BiLSTM.
        Returns:
            A tensor containing the dot product (attention weights between the
            elements of the two sentences).
        """
        attn_weights = K.batch_dot(x=inputs[0],
                                   y=K.permute_dimensions(inputs[1],
                                                          pattern=(0, 2, 1)))
        return K.permute_dimensions(attn_weights, (0, 2, 1))

    @staticmethod
    def _attention_output_shape(inputs):
        input_shape = inputs[0]
        embedding_size = input_shape[1]
        return input_shape[0], embedding_size, embedding_size

    @staticmethod
    def _soft_alignment(inputs):
        """
        Compute the soft alignment between the elements of two sentences.
        Args:
            inputs: A list of two elements, the first is a tensor of attention
                    weights, the second is the encoded sentence on which to
                    compute the alignments.
        Returns:
            A tensor containing the alignments.
        """
        attention = inputs[0]
        sentence = inputs[1]

        # Subtract the max. from the attention weights to avoid overflows.
        exp = K.exp(attention - K.max(attention, axis=-1, keepdims=True))
        exp_sum = K.sum(exp, axis=-1, keepdims=True)
        softmax = exp / exp_sum

        return K.batch_dot(softmax, sentence)

    @staticmethod
    def _soft_alignment_output_shape(inputs):
        attention_shape = inputs[0]
        sentence_shape = inputs[1]
        return attention_shape[0], attention_shape[1], sentence_shape[2]


class SelfAttention(Layer):
    def __init__(self, add_pos=True):
        """Self Attention.
        :return:
        """
        super(SelfAttention, self).__init__()
        self.add_pos = add_pos

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.W = self.add_weight(
            shape=[self.dim, self.dim],
            name='att_weights',
            initializer='random_normal')

    def call(self, inputs, **kwargs):
        q, k, v, mask = inputs
        # pos encoding
        if self.add_pos:
            k += self.positional_encoding(k)
            q += self.positional_encoding(q)
        # Nonlinear transformation
        q = tf.nn.relu(tf.matmul(q, self.W))  # (None, seq_len, dim)
        k = tf.nn.relu(tf.matmul(k, self.W))  # (None, seq_len, dim)
        mat_qk = tf.matmul(q, k, transpose_b=True)  # (None, seq_len, seq_len)
        dk = tf.cast(self.dim, dtype=tf.float32)
        # Scaled
        scaled_att_logits = mat_qk / tf.sqrt(dk)
        # Mask
        mask = tf.tile(mask, [1, 1, q.shape[1]])  # (None, seq_len, seq_len)
        paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask, 0), paddings, scaled_att_logits)  # (None, seq_len, seq_len)
        # softmax
        outputs = tf.nn.softmax(logits=outputs, axis=-1)  # (None, seq_len, seq_len)
        # output
        outputs = tf.matmul(outputs, v)  # (None, seq_len, dim)
        outputs = tf.reduce_mean(outputs, axis=1)  # (None, dim)
        return outputs

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, QK_input):
        angle_rads = self.get_angles(np.arange(QK_input.shape[1])[:, np.newaxis],
                                np.arange(self.dim)[np.newaxis, :], self.dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        """Multi Head Attention Mechanism.
        Args:
            :param d_model: A scalar. The self-attention hidden size.
            :param num_heads: A scalar. Number of heads. If num_heads == 1, the layer is a single self-attention layer.
        :return:
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.wq = Dense(d_model, activation=None)
        self.wk = Dense(d_model, activation=None)
        self.wv = Dense(d_model, activation=None)

    def call(self, q, k, v, mask):
        q = self.wq(q)  # (None, seq_len, d_model)
        k = self.wk(k)  # (None, seq_len, d_model)
        v = self.wv(v)  # (None, seq_len, d_model)
        # split d_model into num_heads * depth
        seq_len, d_model = q.shape[1], q.shape[2]
        q = split_heads(q, seq_len, self.num_heads, q.shape[2] // self.num_heads)  # (None, num_heads, seq_len, depth)
        k = split_heads(k, seq_len, self.num_heads, k.shape[2] // self.num_heads)  # (None, num_heads, seq_len, depth)
        v = split_heads(v, seq_len, self.num_heads, v.shape[2] // self.num_heads)  # (None, num_heads, seq_len, depth)
        # mask
        mask = tf.tile(tf.expand_dims(mask, axis=1), [1, self.num_heads, 1, 1])  # (None, num_heads, seq_len, 1)
        # attention
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)  # (None, num_heads, seq_len, d_model // num_heads)
        # reshape
        outputs = tf.reshape(tf.transpose(scaled_attention, [0, 2, 1, 3]), [-1, seq_len, d_model])  # (None, seq_len, d_model)
        return outputs


class ItemSimilarityGating(Layer):
    def __init__(self, dnn_dropout=0.):
        """Item_similarity_gating, FISSA
        Args:
            dnn_dropout: A scalar.
        :return:
        """
        self.dropout = Dropout(dnn_dropout)
        super(ItemSimilarityGating, self).__init__()

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.W = self.add_weight(
            shape=[3 * self.dim, 1],
            name='att_weights',
            initializer='random_normal')

    def call(self, inputs, **kwargs):
        item_embed, global_info, candidate_embed = inputs
        inputs = tf.concat([item_embed, global_info, candidate_embed], -1)
        inputs = self.dropout(inputs)

        logits = tf.matmul(inputs, self.W)  # (None, neg_num + 1, 1)
        weights = tf.nn.sigmoid(logits)
        return weights
