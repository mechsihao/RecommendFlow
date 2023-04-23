import tensorflow as tf


def qa_auc(y_true, query, doc):
    y_pred = tf.reduce_sum(query * doc, axis=1)
    return tf.keras.metrics.AUC(num_thresholds=100)(y_true, y_pred)


def auc(y_true, y_pred):
    return tf.keras.metrics.AUC(num_thresholds=100)(y_true, y_pred)
