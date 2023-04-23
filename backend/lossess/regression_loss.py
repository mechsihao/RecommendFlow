import tensorflow as tf
import tensorflow.keras.backend as K


def mean_relative_percentage_error(y_true, y_score):
    err = K.abs((y_true - y_score) / K.clip(K.abs(y_true), K.epsilon(), None))
    diff = 1.0 - err
    return 100. * K.mean(diff, axis=-1)