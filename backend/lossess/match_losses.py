import tensorflow as tf
from tensorflow.keras import backend as K


@tf.function
def mean_squared_error(y_true, query, doc):
    """均方差核心函数
    """
    y_pred = tf.reduce_sum(query * doc, axis=1)
    return tf.reduce_mean((y_true - y_pred) ** 2)


@tf.function
def binary_cross_entropy(y_true, query, doc):
    """交叉熵核心函数
    """
    y_pred = tf.reduce_sum(query * doc, axis=1)
    return K.binary_crossentropy(y_true, y_pred)


@tf.function
def cosent_loss(y_true, y_pred, scale=20):
    """cosent 核心函数
    """
    y_true = K.cast(y_true[:, None] < y_true[None, :], K.floatx())  # query-doc对 label 两两比较，y_true[i, j]代表 第i对label 是否小于 第j对label
    y_pred = y_pred * scale  # query-doc 之间的相似度
    y_pred = y_pred[:, None] - y_pred[None, :]  # query-doc对 之间的相似度的差值，y_pred[i, j]代表 第i对预测分数 和 第j对预测分数 之间的差值
    y_pred = K.reshape(y_pred - tf.cast((1 - y_true) * 1e12, tf.float32), [-1])
    # loss函数的最终计算为 reduce_logsumexp = log(∑(exp(x_i)))
    # 1e12可以看成∞，前面加个减号变成-∞，由于exp(-∞)=0，也就是说y_pred中 y_true为1所对应的地方才是最终对loss有贡献的地方，其余地方都是-∞，对应loss为0
    # 但这里有个比较难以理解的地方，因为我们期望正样本对的相似度 大于 负样本对，如果y_true[i, j] = 1,则代表 label_i < label_j，那么此时如果 pred_i < pred_j,应
    # 该是符合预期的，此时loss应该为0才对。
    # 但是用cosent计算并不是0，而是exp(pred_i - pred_j)，虽然这个分数比较小，但是仍然是有loss的，并且由于scale的加入，loss还会被进一步放大。
    # 为了修改这个问题，可以将y_pred 为负数的部分也置为-1e12，这样就可以同样忽略 满足期望的部分 loss，将其置为零，详细实现方案见cosent_loss_v2
    y_pred = K.concatenate([[0], y_pred], axis=0)  # 这个是为了对应公式最前面的+1，保证loss都是大于0的
    return tf.reduce_logsumexp(y_pred, axis=None, keepdims=False)  # log(∑(exp(x_i)))


@tf.function
def cosent_loss(y_true, query, doc, scale=20):
    """cosent 核心函数
    """
    y_true = K.cast(y_true[:, None] < y_true[None, :], K.floatx())  # query-doc对 label 两两比较，y_true[i, j]代表 第i对label 是否小于 第j对label
    y_pred = K.sum(query * doc, axis=1) * scale  # query-doc 之间的相似度
    y_pred = y_pred[:, None] - y_pred[None, :]  # query-doc对 之间的相似度的差值，y_pred[i, j]代表 第i对预测分数 和 第j对预测分数 之间的差值
    y_pred = K.reshape(y_pred - tf.cast((1 - y_true) * 1e12, tf.float32), [-1])
    # loss函数的最终计算为 reduce_logsumexp = log(∑(exp(x_i)))
    # 1e12可以看成∞，前面加个减号变成-∞，由于exp(-∞)=0，也就是说y_pred中 y_true为1所对应的地方才是最终对loss有贡献的地方，其余地方都是-∞，对应loss为0
    # 但这里有个比较难以理解的地方，因为我们期望正样本对的相似度 大于 负样本对，如果y_true[i, j] = 1,则代表 label_i < label_j，那么此时如果 pred_i < pred_j,应
    # 该是符合预期的，此时loss应该为0才对。
    # 但是用cosent计算并不是0，而是exp(pred_i - pred_j)，虽然这个分数比较小，但是仍然是有loss的，并且由于scale的加入，loss还会被进一步放大。
    # 为了修改这个问题，可以将y_pred 为负数的部分也置为-1e12，这样就可以同样忽略 满足期望的部分 loss，将其置为零，详细实现方案见cosent_loss_v2
    y_pred = K.concatenate([[0], y_pred], axis=0)  # 这个是为了对应公式最前面的+1
    return tf.reduce_logsumexp(y_pred, axis=None, keepdims=False)  # log(∑(exp(x_i)))


@tf.function
def cosent_loss_v2(y_true, query, doc, scale=20):
    """相比于cosent，将负数部分也置为-1e12，使其对整体loss无贡献
    """
    y_true = K.cast(y_true[:, None] < y_true[None, :], tf.float64)
    y_pred = K.sum(query * doc, axis=1) * scale
    y_pred = y_pred[:, None] - y_pred[None, :]
    y_pred = K.reshape(y_pred - (1 - y_true) * 1e12, [-1])
    y_pred = tf.where(y_pred > 0, y_pred, -1e12)  # 最核心的改变，将负数部分全部置为-1e12，使其对整体loss无贡献
    y_pred = K.concatenate([[0], y_pred], axis=0)
    return tf.reduce_logsumexp(y_pred, axis=None, keepdims=False)


@tf.function
def aux_label_cosent_loss(y_true, aux_true, query, doc, scale=20, alpha: float = .5):
    """
    辅助指标的cosent，该loss会用正样本和负样本来分别对辅助label做cosent最后求和
    比较适合的辅助指标有：
        - 出价bid（也可以是bid分箱后再对比，这里要注意，必须得让bid本身是可比的，比如可以让大家都换算成ocpc bid，越深度转化bid越高，因此需要统一转化下）
        - ecpm = bid * ctr
        - 其它业务指标
    因此该loss只要是想让某些业务指标排序靠前的时候都可以使用
    :param alpha:
    :param y_true: 0/1 label
    :param aux_true: 辅助label
    :param query: query向量
    :param doc: doc向量
    :param scale: 缩放参数
    :param alpha: 负样本辅助label权重
    :return: loss值
    """
    pos_ind = tf.squeeze(tf.where(y_true == 1))  # 这里其实考虑了全部都为0的情况，loss返回直接是0
    neg_ind = tf.squeeze(tf.where(y_true == 0))  # 这里其实考虑了全部都为0的情况，loss返回直接是0
    pos_loss = cosent_loss_v2(tf.gather(aux_true, pos_ind), tf.gather(query, pos_ind), tf.gather(doc, pos_ind), scale)
    neg_loss = cosent_loss_v2(tf.gather(aux_true, neg_ind), tf.gather(query, neg_ind), tf.gather(doc, neg_ind), scale)
    return (1 - alpha) * pos_loss + alpha * neg_loss


@tf.function
def pos_aux_label_cosent_loss(y_true, aux_true, query, doc, scale=20):
    """
    辅助指标的cosent，该loss会过滤掉样本中的负样本，仅用正样本来对辅助label做cosent
    比较适合的辅助指标有：
        - 出价bid（也可以是bid分箱后再对比，这里要注意，必须得让bid本身是可比的，比如可以让大家都换算成ocpc bid，越深度转化bid越高，因此需要统一转化下）
        - ecpm = bid * ctr
        - 其它业务指标
    因此该loss只要是想让某些业务指标排序靠前的时候都可以使用
    :param y_true: 0/1 label
    :param aux_true: 辅助label
    :param query: query向量
    :param doc: doc向量
    :param scale: 缩放参数
    :return: loss值
    """
    pos_ind = tf.squeeze(tf.where(y_true == 1))  # 这里其实考虑了全部都为0的情况，loss返回直接是0
    return cosent_loss_v2(tf.gather(aux_true, pos_ind), tf.gather(query, pos_ind), tf.gather(doc, pos_ind), scale)


@tf.function
def batch_neg_sample_ce_loss(y_true, query, doc):
    """
    batch 内负采样交叉熵，自实现，相当于query是logits，doc是label，即分类query，ground-truth为doc，query对doc的多分类交叉熵
    y_true: 标签/打分 [2 * None, 1]
    y_pred: 句向量 [2 * None, 768]
    scale: 温度参数
    实现batch内负采样，公式为：
    loss𝑖 = −(1/n) * Sigma(y_true·log(y_pred) + (1-y_true)·log(1-y_pred))
    """
    y_true = tf.linalg.diag(y_true)
    y_pred = tf.matmul(query, tf.transpose(doc))
    return tf.reduce_mean(K.categorical_crossentropy(y_true, y_pred) * tf.linalg.diag_part(y_true))


@tf.function
def batch_neg_sample_symmetrical_ce_loss(y_true, query, doc):
    """
    batch 内负采样交叉熵，自实现，包含对称部分，增加了doc对query的多分类交叉熵
    y_true: 标签/打分 [2 * None, 1]
    y_pred: 句向量 [2 * None, 768]
    scale: 温度参数
    实现batch内负采样，公式为：
    """
    y_true = tf.linalg.diag(y_true)
    y_pred1 = tf.matmul(query, tf.transpose(doc))
    y_pred2 = tf.matmul(doc, tf.transpose(query))
    loss = 1/2 * (K.categorical_crossentropy(y_true, y_pred1) + K.categorical_crossentropy(y_true, y_pred2)) * tf.linalg.diag_part(y_true)
    return tf.reduce_mean(loss)


@tf.function
def batch_neg_sample_scaled_multi_class_ce_loss(y_true, query, doc, scale=20):
    """
    batch 内负采样交叉熵
    复现自论文：Que2Search: https://zhuanlan.zhihu.com/p/415516966
    y_true: 标签/打分 [2 * None, 1], 不同于论文中的loss只允许label为1，这里允许label可以有0也有1，目的是加入加入热门物料作为负样本，进行热度降权
    y_pred: 句向量 [2 * None, 768]
    scale: 温度参数
    实现batch内负采样，公式为：
        - loss𝑖 = −log(exp(𝑠·cos{𝑞𝑖,𝑑𝑖})/ Sigma(exp(𝑠·cos{𝑞𝑖,𝑑𝑗})))
    """
    y_true = tf.linalg.diag(y_true)
    y_pred = tf.matmul(query, tf.transpose(doc))
    num = tf.linalg.diag_part(tf.exp(scale * y_pred))  # 只取对角线上的exp值
    den = tf.reduce_sum(tf.exp(scale * y_pred), axis=-1)
    loss = -K.log(num / den) * tf.linalg.diag_part(y_true)
    return tf.reduce_mean(loss)


@tf.function
def batch_neg_sample_symmetrical_scaled_multi_class_ce_loss(y_true, query, doc, scale=20):
    """
    复现自论文：Que2Search: https://zhuanlan.zhihu.com/p/415516966
    batch 内负采样交叉熵
    y_true: 标签/打分 [2 * None, 1], 不同于论文中的loss只允许label为1，这里允许label可以有0也有1，目的是加入加入热门物料作为负样本，进行热度降权
    y_pred: 句向量 [2 * None, 768]
    scale: 温度参数
    实现batch内负采样，公式为：
    loss𝑖 = −1/2*[log(exp(𝑠·cos{𝑞𝑖,𝑑𝑖})/ Sigma(exp(𝑠·cos{𝑞𝑖,𝑑𝑗}))) + log(exp(𝑠·cos{𝑞𝑖,𝑑𝑖})/ Sigma(exp(𝑠·cos{𝑞𝑗,𝑑𝑖})))]
    """
    y_true = tf.linalg.diag(y_true)
    y_pred = scale * tf.matmul(query, tf.transpose(doc))
    # 相当于对query做分类任务，类别是doc
    num1 = tf.linalg.diag_part(tf.exp(scale * y_pred))  # 只取对角线上的exp值
    den1 = tf.reduce_sum(tf.exp(scale * y_pred), axis=-1)
    # 相当于对doc做分类任务，类别是query
    num2 = tf.linalg.diag_part(tf.exp(scale * y_pred))  # 只取对角线上的exp值
    den2 = tf.reduce_sum(tf.exp(scale * y_pred), axis=-1)
    # 最后将两部分加和，很明显，不光照顾了query的分类准确度，还照顾到了doc侧的分类准确度，在doc为主的场景更有效。
    loss = -1/2 * (K.log(num1 / den1) + K.log(num2 / den2)) * tf.linalg.diag_part(y_true)
    return tf.reduce_mean(loss)


@tf.function
def batch_neg_sample_margin_rank_loss(y_true, query, doc, margin=0.1):
    """
    自实现batch内负采样的ltr loss，每个query除了正样本以外其他doc均为负样本
    y_true: 标签/打分 [2 * None, 1]
    y_pred: 句向量 [2 * None, 768]
    margin: 间隔参数，一般取0.1 ~ 0.2最佳
    loss𝑖 = Sigma_j(𝑚𝑎𝑥(0, −[𝑐𝑜𝑠(𝑞𝑖,𝑑𝑖) − 𝑐𝑜𝑠(𝑞𝑖,𝑑𝑛𝑞𝑖_j)] + 𝑚𝑎𝑟𝑔𝑖𝑛))
    """
    y_pred = tf.matmul(query, tf.transpose(doc))
    y_sub = -(tf.linalg.diag_part(y_pred)[:, None] - y_pred) + margin

    loss = tf.clip_by_value(y_sub, 0, 1e14) * y_true
    return tf.reduce_sum(loss)


@tf.function
def batch_hard_neg_sample_margin_rank_loss(y_true, query, doc, margin=0.1):
    """
    复现自论文：Que2Search: https://zhuanlan.zhihu.com/p/415516966
    batch 内负采样交叉熵，并且获取除了正样本外相似度最高的一个样本为困难负样本
    注意：该Loss不能在随机初始化的模型上使用！因为需要用相似度来评估出困难样本，因此需要模型首先收敛，其次再进行困难训练
    y_true: 标签/打分 [2 * None, 1]
    y_pred: 句向量 [2 * None, 768]
    margin: 间隔参数，一般取0.1 ~ 0.2最佳
    """
    y_pred = tf.matmul(query, tf.transpose(doc))
    y_pos_cos = tf.linalg.diag_part(y_pred)

    y_neg_pred = y_pred - tf.linalg.diag(tf.linalg.diag_part(y_pred))  # 将y_pred的对角线都置为0
    y_neg_cos = K.max(y_neg_pred, axis=-1)
    y_sub = -(y_pos_cos - y_neg_cos) + margin

    loss = tf.clip_by_value(y_sub, 0, 1e14) * y_true
    return tf.reduce_sum(loss)
