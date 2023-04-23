import tensorflow as tf


def get_devices():
    return tf.config.list_physical_devices(device_type='GPU')


def set_mem_growth_gpus():
    for gpu in get_devices():
        tf.config.experimental.set_memory_growth(gpu, True)


def auto_device_strategy():
    return tf.distribute.MirroredStrategy()
