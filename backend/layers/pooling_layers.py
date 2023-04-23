import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec


class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=2)
        self.k = k

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.k

    def call(self, inputs, *args, **kwargs):
        top_k = tf.nn.top_k(inputs, k=self.k, sorted=True, name=None)[0]
        return top_k

    def get_config(self):
        base_config = super(KMaxPooling, self).get_config()
        config = {'k': self.k}
        base_config.update(config)
        return base_config


class DynamicPoolingLayer(Layer):
    """
    Layer that computes dynamic pooling of one tensor.
    :param psize1: pooling size of dimension 1
    :param psize2: pooling size of dimension 2
    :param kwargs: Standard layer keyword arguments.
    """
    def __init__(self, psize1, psize2, **kwargs):
        super().__init__(**kwargs)
        self._psize1 = psize1
        self._psize2 = psize2
        self._msize1 = None
        self._msize2 = None

    def build(self, input_shape):
        super().build(input_shape)
        input_shape_one = input_shape[0]
        self._msize1 = input_shape_one[1]
        self._msize2 = input_shape_one[2]

    def call(self, inputs, **kwargs):
        """
        The computation logic of DynamicPoolingLayer.
        :param inputs: two input tensors.
        """
        self._validate_dpool_size()
        x, dpool_index = inputs
        dpool_shape = tf.shape(dpool_index)
        batch_index_one = tf.expand_dims(
            tf.expand_dims(
                tf.range(dpool_shape[0]), axis=-1),
            axis=-1)
        batch_index = tf.expand_dims(
            tf.tile(batch_index_one, [1, self._msize1, self._msize2]),
            axis=-1)
        dpool_index_ex = tf.concat([batch_index, dpool_index], axis=3)
        x_expand = tf.gather_nd(x, dpool_index_ex)
        stride1 = self._msize1 // self._psize1
        stride2 = self._msize2 // self._psize2

        x_pool = tf.nn.max_pool(x_expand,
                                [1, stride1, stride2, 1],
                                [1, stride1, stride2, 1],
                                "VALID")
        return x_pool

    def compute_output_shape(self, input_shape):
        """
        Calculate the layer output shape.
        :param input_shape: the shapes of the input tensors,
            for DynamicPoolingLayer we need tow input tensors.
        """
        input_shape_one = input_shape[0]
        return None, self._psize1, self._psize2, input_shape_one[3]

    def get_config(self) -> dict:
        """Get the config dict of DynamicPoolingLayer."""
        base_config = super(DynamicPoolingLayer, self).get_config()
        config = {
            'psize1': self._psize1,
            'psize2': self._psize2
        }
        base_config.update(config)
        return base_config

    def _validate_dpool_size(self):
        suggestion = self.get_size_suggestion(
            self._msize1, self._msize2, self._psize1, self._psize2
        )
        if suggestion != (self._psize1, self._psize2):
            raise ValueError(
                "DynamicPooling Layer can not "
                f"generate ({self._psize1} x {self._psize2}) output "
                f"feature map, please use ({suggestion[0]} x {suggestion[1]})"
                f" instead. `model.params['dpool_size'] = {suggestion}` "
            )

    @classmethod
    def get_size_suggestion(cls, msize1, msize2, psize1, psize2):
        """
        Get `dpool_size` suggestion for a given shape.
        Returns the nearest legal `dpool_size` for the given combination of
        `(psize1, psize2)`.
        :param msize1: size of the left text.
        :param msize2: size of the right text.
        :param psize1: base size of the pool.
        :param psize2: base size of the pool.
        :return:
        """
        stride1 = msize1 // psize1
        stride2 = msize2 // psize2
        suggestion1 = msize1 // stride1
        suggestion2 = msize2 // stride2

        return suggestion1, suggestion2
