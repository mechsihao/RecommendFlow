import tensorflow as tf


def create_mlp(hidden_units, dropout_rate, activation, normalization_layer, name=None):
    """
    根据配置来生成MLP模块，每一层包含 [norm_layer, dense, dropout]
    :return: Sequential
    """
    mlp_layers = []
    for units in hidden_units:
        mlp_layers.append(normalization_layer)
        mlp_layers.append(tf.keras.layers.Dense(units, activation=activation))
        mlp_layers.append(tf.keras.layers.Dropout(dropout_rate))

    return tf.keras.Sequential(mlp_layers, name=name)


def bn_layer(bn_in, training, name="bn", **kv):
    layer = tf.keras.layers.BatchNormalization(name=name, **kv)
    res = layer.apply(bn_in, training=training)
    return res


def dice_func(pre_active, training, name):
    alphas = tf.compat.v1.get_variable(
        "alpha_" + name, pre_active.get_shape()[-1], initializer=tf.compat.v1.constant_initializer(0.0), dtype=pre_active.dtype.base_dtype)
    bn_out = bn_layer(pre_active, training, name="bn_" + name, center=False, scale=False)
    sig = tf.sigmoid(bn_out)
    return alphas * (1.0 - sig) * pre_active + sig * pre_active
