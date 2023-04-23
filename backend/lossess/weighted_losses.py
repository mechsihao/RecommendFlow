import tensorflow as tf
import tensorflow.keras.backend as K


@tf.function
def mean_squared_error_weighted(y_true, query, doc):
    """均方差核心函数
    """
    y_pred = tf.reduce_sum(query * doc, axis=1)
    return tf.reduce_mean((y_true - y_pred) ** 2)


@tf.function
def binary_cross_entropy_weighted(y_true, query, doc):
    """交叉熵核心函数
    """
    y_pred = tf.reduce_sum(query * doc, axis=1)
    return tf.losses.binary_crossentropy(y_true, y_pred)


@tf.function
def cosent_loss_weighted(y_true, query, doc, scale=20):
    """cosent 核心函数
    """
    y_true = K.cast(y_true[:, None] < y_true[None, :], K.floatx())
    y_pred = K.sum(query * doc, axis=1) * scale  # 计算句子对之间的相似度 y_pred [None, ]，也可以直接加 0.2 ~ 0.4
    y_pred = y_pred[:, None] - y_pred[None, :]  # 矩阵y_pred[i, j] 代表第 i个句子对的cos值 和 第j个句子对cos值 的差值，y_pred [None, None]
    y_pred = K.reshape(y_pred - tf.cast((1 - y_true) * 1e12, tf.float32), [-1])
    y_pred = K.concatenate([[0], y_pred], axis=0)
    return tf.reduce_logsumexp(y_pred, axis=None, keepdims=False)  # K.logsumexp
