import tensorflow as tf
from tensorflow.keras import backend as K
from backend.losses import match_losses


@tf.function
def zip_embedding(q, a):
    """
    将query、doc打包成一个向量，使之可以传入loss中
    :param q: query [n * embedding_dim]
    :param a: doc [n * embedding_dim]
    :return: 整合成一个embedding [2n * (embedding_dim + 1)]
    """
    t = tf.keras.layers.concatenate([q, a], axis=1)
    return tf.reshape(t, (-1, a.shape[1]))


@tf.function
def unzip_embedding(y_true, y_pred):
    """
    从keras loss输入中获取query、doc、label
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = tf.squeeze(y_true, axis=1)
    query, doc = K.l2_normalize(y_pred[::2], axis=1), K.l2_normalize(y_pred[1::2], axis=1)
    return tf.cast(y_true, tf.float64), tf.cast(query, tf.float64), tf.cast(doc, tf.float64)


@tf.function
def mean_squared_error(y_true, y_pred):
    """均方差
    """
    y_true, query, doc = unzip_embedding(y_true, y_pred)
    return match_losses.mean_squared_error(y_true, query, doc)


@tf.function
def binary_cross_entropy(y_true, y_pred):
    """交叉熵
    """
    y_true, query, doc = unzip_embedding(y_true, y_pred)
    return match_losses.binary_cross_entropy(y_true, query, doc)


@tf.function
def cosent_loss(y_true, y_pred, scale=20):
    y_true, query, doc = unzip_embedding(y_true, y_pred)
    return match_losses.cosent_loss(y_true, query, doc, scale)


@tf.function
def batch_neg_sample_ce_loss(y_true, y_pred):
    """
    batch 内负采样交叉熵，自实现，相当于query是logits，doc是label，即分类query，ground-truth为doc，query对doc的多分类交叉熵
    y_true: 标签/打分 [2 * None, 1]
    y_pred: 句向量 [2 * None, 768]
    scale: 温度参数
    实现batch内负采样，公式为：
    loss𝑖 = −(1/n) * Sigma(y_true·log(y_pred) + (1-y_true)·log(1-y_pred))
    """
    y_true, query, doc = unzip_embedding(y_true, y_pred)
    return match_losses.batch_neg_sample_ce_loss(y_true, query, doc)


@tf.function
def batch_neg_sample_symmetrical_ce_loss(y_true, y_pred):
    """
    batch 内负采样交叉熵，自实现，包含对称部分，增加了doc对query的多分类交叉熵
    y_true: 标签/打分 [2 * None, 1]
    y_pred: 句向量 [2 * None, 768]
    scale: 温度参数
    实现batch内负采样，公式为：
    """
    y_true, query, doc = unzip_embedding(y_true, y_pred)
    return match_losses.batch_neg_sample_symmetrical_ce_loss(y_true, query, doc)


@tf.function
def batch_neg_sample_scaled_multi_class_ce_loss(y_true, y_pred, scale=20):
    """
    batch 内负采样交叉熵
    复现自论文：Que2Search: https://zhuanlan.zhihu.com/p/415516966
    y_true: 标签/打分 [2 * None, 1], 不同于论文中的loss只允许label为1，这里允许label可以有0也有1，目的是加入加入热门物料作为负样本，进行热度降权
    y_pred: 句向量 [2 * None, 768]
    scale: 温度参数
    实现batch内负采样，公式为：
        - loss𝑖 = −log(exp(𝑠·cos{𝑞𝑖,𝑑𝑖})/ Sigma(exp(𝑠·cos{𝑞𝑖,𝑑𝑗})))
    """
    y_true, query, doc = unzip_embedding(y_true, y_pred)
    return match_losses.batch_neg_sample_scaled_multi_class_ce_loss(y_true, query, doc, scale)


@tf.function
def batch_neg_sample_symmetrical_scaled_multi_class_ce_loss(y_true, y_pred, scale=20):
    """
    复现自论文：Que2Search: https://zhuanlan.zhihu.com/p/415516966
    batch 内负采样交叉熵
    y_true: 标签/打分 [2 * None, 1], 不同于论文中的loss只允许label为1，这里允许label可以有0也有1，目的是加入加入热门物料作为负样本，进行热度降权
    y_pred: 句向量 [2 * None, 768]
    scale: 温度参数
    实现batch内负采样，公式为：
    loss𝑖 = −1/2*[log(exp(𝑠·cos{𝑞𝑖,𝑑𝑖})/ Sigma(exp(𝑠·cos{𝑞𝑖,𝑑𝑗}))) + log(exp(𝑠·cos{𝑞𝑖,𝑑𝑖})/ Sigma(exp(𝑠·cos{𝑞𝑗,𝑑𝑖})))]
    """
    y_true, query, doc = unzip_embedding(y_true, y_pred)
    return match_losses.batch_neg_sample_symmetrical_scaled_multi_class_ce_loss(y_true, query, doc, scale)


@tf.function
def batch_neg_sample_margin_rank_loss(y_true, y_pred, margin=0.1):
    """
    自实现batch内负采样的ltr loss，每个query除了正样本以外其他doc均为负样本
    y_true: 标签/打分 [2 * None, 1]
    y_pred: 句向量 [2 * None, 768]
    margin: 间隔参数，一般取0.1 ~ 0.2最佳
    loss𝑖 = Sigma_j(𝑚𝑎𝑥(0, −[𝑐𝑜𝑠(𝑞𝑖,𝑑𝑖) − 𝑐𝑜𝑠(𝑞𝑖,𝑑𝑛𝑞𝑖_j)] + 𝑚𝑎𝑟𝑔𝑖𝑛))
    """
    y_true, query, doc = unzip_embedding(y_true, y_pred)
    return match_losses.batch_neg_sample_margin_rank_loss(y_true, query, doc, margin)


@tf.function
def batch_hard_neg_sample_margin_rank_loss(y_true, y_pred, margin=0.1):
    """
    复现自论文：Que2Search: https://zhuanlan.zhihu.com/p/415516966
    batch 内负采样交叉熵，并且获取除了正样本外相似度最高的一个样本为困难负样本
    注意：该Loss不能在随机初始化的模型上使用！因为需要用相似度来评估出困难样本，因此需要模型首先收敛，其次再进行困难训练
    y_true: 标签/打分 [2 * None, 1]
    y_pred: 句向量 [2 * None, 768]
    margin: 间隔参数，一般取0.1 ~ 0.2最佳
    """
    y_true, query, doc = unzip_embedding(y_true, y_pred)
    return match_losses.batch_hard_neg_sample_margin_rank_loss(y_true, query, doc, margin)
