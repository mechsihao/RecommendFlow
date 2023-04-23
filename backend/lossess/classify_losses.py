import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, CategoricalCrossentropy, CategoricalHinge


@tf.function
def multilabel_categorical_crossentropy(y_true, y_pred):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax！预测阶段则输出y_pred大于0的类。
         如有疑问，请仔细阅读并理解本文：https://kexue.fm/archives/7359
    """
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred_neg = K.concatenate([y_pred_neg, zeros], axis=-1)
    y_pred_pos = K.concatenate([y_pred_pos, zeros], axis=-1)
    neg_loss = K.logsumexp(y_pred_neg, axis=-1)
    pos_loss = K.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


@tf.function
def sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=False):
    """稀疏版多标签分类的交叉熵
    说明：
        1. y_true.shape=[..., num_positive]，
           y_pred.shape=[..., num_classes]；
        2. 请保证y_pred的值域是全体实数，换言之一般情况下y_pred不用加激活函数，尤其是不能加sigmoid或者softmax；
        3. 预测阶段则输出y_pred大于0的类；
        4. 详情请看：https://kexue.fm/archives/7359
    """
    zeros = K.zeros_like(y_pred[..., :1])
    y_pred = K.concatenate([y_pred, zeros], axis=-1)
    inf_vecs = zeros + 1e12

    if mask_zero:
        y_pred = K.concatenate([inf_vecs, y_pred[..., 1:]], axis=-1)

    y_pos_2 = tf.gather(y_pred, y_true, batch_dims=K.ndim(y_true)-1)
    y_pos_1 = K.concatenate([y_pos_2, zeros], axis=-1)
    if mask_zero:
        y_pred = K.concatenate([-inf_vecs, y_pred[..., 1:]], axis=-1)
        y_pos_2 = tf.gather(y_pred, y_true, batch_dims=K.ndim(y_true)-1)
    pos_loss = K.logsumexp(-y_pos_1, axis=-1)
    all_loss = K.logsumexp(y_pred, axis=-1)
    aux_loss = K.logsumexp(y_pos_2, axis=-1) - all_loss
    aux_loss = K.clip(1 - K.exp(aux_loss), K.epsilon(), 1)
    neg_loss = all_loss + K.log(aux_loss)
    return pos_loss + neg_loss


@tf.function
def sparse_categorical_crossentropy(y_true, y_pred):
    return SparseCategoricalCrossentropy(y_true, y_pred)


@tf.function
def binary_crossentropy(y_true, y_pred):
    return BinaryCrossentropy(y_true, y_pred)


@tf.function
def category_crossentropy(y_true, y_pred):
    return CategoricalCrossentropy(y_true, y_pred)


@tf.function
def category_hinge(y_true, y_pred):
    return CategoricalHinge(y_true, y_pred)


def binary_focal_loss(y_true, y_score, gamma=2., alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
    p_t = y_true * y_score + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_score) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
    return K.mean(focal_loss)


def categorical_ghm_loss(bins=30, momentum=0.75):
    """ 返回多分类 GHM 损失函数：把每个区间上的梯度做平均，也就是说把梯度拉平，回推到公式上等价于把loss做平均
    Formula:
        loss_i = sum(crossentropy_loss(p_i,p*_i) / GD(g_i))
        GD(g) = S_ind(g) / delta = S_ind(g) * M
        S_ind(g) = momentum * S_ind(g) + (1 - momentum) * R_ind(g)
        R_ind(g)是 g=|p-p*| 所在梯度区间[(i-1)*delta, i*delta]的样本数
        M = 1/delta，这个是个常数，理论上去掉只有步长影响
    Parameters: （论文默认）
        bins -- 区间个数，default 30
        momentum -- 使用移动平均来求区间内样本数，动量部分系数，论文说不敏感
    """
    # 区间边界
    edges = np.array([i / bins for i in range(bins + 1)])
    edges = np.expand_dims(np.expand_dims(edges, axis=-1), axis=-1)
    acc_sum = 0
    if momentum > 0:
        acc_sum = tf.zeros(shape=(bins,), dtype=tf.float32)

    def ghm_class_loss(y_truth, y_pred, valid_mask):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # 0. 计算本次mini-batch的梯度分布：R_ind(g)
        gradient = K.abs(y_truth - y_pred)
        # 获取概率最大的类别下标，将该类别的梯度做为该标签的梯度代表
        # 没有这部分就是每个类别的梯度都参与到GHM，实验表明没有这部分会更好些
        # truth_indices_1 = K.expand_dims(K.argmax(y_truth, axis=1))
        # truth_indices_0 = K.expand_dims(K.arange(start=0, stop=tf.shape(y_pred)[0], step=1, dtype='int64'))
        # truth_indices = K.concatenate([truth_indices_0, truth_indices_1])
        # main_gradient = tf.gather_nd(gradient, truth_indices)
        # gradient = tf.tile(tf.expand_dims(main_gradient, axis=-1), [1, y_pred.shape[1]])

        # 求解各个梯度所在的区间，并落到对应区间内进行密度计数
        grads_bin = tf.logical_and(tf.greater_equal(gradient, edges[:-1, :, :]), tf.less(gradient, edges[1:, :, :]))
        valid_bin = tf.boolean_mask(grads_bin, valid_mask, name='valid_gradient', axis=1)
        valid_bin = tf.reduce_sum(tf.cast(valid_bin, dtype=tf.float32), axis=(1, 2))
        # 2. 更新指数移动平均后的梯度分布：S_ind(g)
        nonlocal acc_sum
        acc_sum = tf.add(momentum * acc_sum, (1 - momentum) * valid_bin, name='update_bin_number')
        # sample_num = tf.reduce_sum(acc_sum)  # 是否乘以总数，乘上效果反而变差了
        # 3. 计算本次mini-batch不同loss对应的梯度密度：GD(g)
        position = tf.slice(tf.where(grads_bin), [0, 1], [-1, 2])
        value = tf.gather_nd(acc_sum, tf.slice(tf.where(grads_bin), [0, 0], [-1, 1]))  # * bins
        grad_density = tf.sparse.SparseTensor(indices=position, values=value, dense_shape=tf.shape(gradient, out_type=tf.int64))
        grad_density = tf.sparse.to_dense(grad_density, validate_indices=False)
        grad_density = grad_density * tf.expand_dims(valid_mask, -1) + (1 - tf.expand_dims(valid_mask, -1))

        # 4. 计算本次mini-batch不同样本的损失：loss
        cross_entropy = -y_truth * K.log(y_pred)
        # loss = cross_entropy / grad_density * sample_num
        loss = cross_entropy / grad_density
        return K.sum(loss, axis=1)

    return ghm_class_loss


def categorical_focal_loss(gamma=2.0, alpha=1.0):
    """ 返回多分类 focal loss 函数
    Formula: loss = -alpha*((1-p_t)^gamma)*log(p_t)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy, default 0.25
        gamma -- focusing parameter for modulating factor (1-p), default 2.0
    """
    def focal_loss(y_truth, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_truth * K.log(y_pred)
        weight = alpha * K.pow(K.abs(y_truth - y_pred), gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss
