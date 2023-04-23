import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, LayerNormalization, Conv1D, ReLU, MultiHeadAttention
from tensorflow.keras.regularizers import l2

from layer_utils import index_mapping


class FM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=0., v_reg=0.):
        """Factorization Machines.
        Args:
            :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
            :param k: A scalar. The latent vector.
            :param w_reg: A scalar. The regularization coefficient of parameter w.
            :param v_reg: A scalar. The regularization coefficient of parameter v.
        :return:
        """
        super(FM_Layer, self).__init__()
        self.feature_columns = feature_columns
        self.field_num = len(feature_columns)
        self.map_dict = {}
        self.feature_length = 0
        for feat in self.feature_columns:
            self.map_dict[feat['feat_name']] = self.feature_length
            self.feature_length += feat['feat_num']
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.feature_length, self.k),
                                 initializer=tf.random_normal_initializer(),
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs):
        # mapping
        inputs = index_mapping(inputs, self.map_dict)
        inputs = tf.concat([value for _, value in inputs.items()], axis=-1)
        # first order
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # second order
        second_inputs = tf.nn.embedding_lookup(self.V, inputs)  # (batch_size, fields, embed_dim)
        square_sum = tf.square(tf.reduce_sum(second_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(second_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        # outputs
        outputs = first_order + second_order
        return outputs


class FFM_Layer(Layer):
    def __init__(self, feature_columns, k, w_reg=0., v_reg=0.):
        """Field Factorization Machines Layer.
        Args:
            :param feature_columns: A list. [{'feat_name':, 'feat_num':, 'embed_dim':}, ...]
            :param k: A scalar. The latent vector.
            :param w_reg: A scalar. The regularization coefficient of parameter w.
            :param v_reg: A scalar. The regularization coefficient of parameter v.
        :return:
        """
        super(FFM_Layer, self).__init__()
        self.feature_columns = feature_columns
        self.field_num = len(self.feature_columns)
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg
        self.map_dict = {}
        self.feature_length = 0
        for feat in self.feature_columns:
            self.map_dict[feat['feat_name']] = self.feature_length
            self.feature_length += feat['feat_num']

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer='random_normal',
                                 regularizer=l2(self.w_reg),
                                 trainable=True)
        self.v = self.add_weight(name='v',
                                 shape=(self.feature_length, self.field_num, self.k),
                                 initializer='random_normal',
                                 regularizer=l2(self.v_reg),
                                 trainable=True)

    def call(self, inputs):
        inputs = index_mapping(inputs, self.map_dict)
        inputs = tf.concat([value for _, value in inputs.items()], axis=-1)
        # first order
        first_order = self.w0 + tf.reduce_sum(tf.nn.embedding_lookup(self.w, inputs), axis=1)  # (batch_size, 1)
        # field second order
        second_order = 0
        latent_vector = tf.reduce_sum(tf.nn.embedding_lookup(self.v, inputs), axis=1)  # (batch_size, field_num, k)
        for i in range(self.field_num):
            for j in range(i+1, self.field_num):
                second_order += tf.reduce_sum(latent_vector[:, i] * latent_vector[:, j], axis=1, keepdims=True)
        return first_order + second_order


class Residual_Units(Layer):
    def __init__(self, hidden_unit, dim_stack):
        """Residual Units.
        Args:
            :param hidden_unit: A list. Neural network hidden units.
            :param dim_stack: A scalar. The dimension of inputs unit.
        :return:
        """
        super(Residual_Units, self).__init__()
        self.layer1 = Dense(units=hidden_unit, activation='relu')
        self.layer2 = Dense(units=dim_stack, activation=None)
        self.relu = ReLU()

    def call(self, inputs):
        x = inputs
        x = self.layer1(x)
        x = self.layer2(x)
        outputs = self.relu(x + inputs)
        return outputs


class CrossNetwork(Layer):
    def __init__(self, layer_num, reg_w=0., reg_b=0.):
        """CrossNetwork.
        Args:
            :param layer_num: A scalar. The depth of cross network.
            :param reg_w: A scalar. The regularization coefficient of w.
            :param reg_b: A scalar. The regularization coefficient of b.
        :return:
        """
        super(CrossNetwork, self).__init__()
        self.layer_num = layer_num
        self.reg_w = reg_w
        self.reg_b = reg_b

    def build(self, input_shape):
        dim = int(input_shape[-1])
        self.cross_weights = [
            self.add_weight(name='w_' + str(i),
                            shape=(dim, 1),
                            initializer='random_normal',
                            regularizer=l2(self.reg_w),
                            trainable=True
                            )
            for i in range(self.layer_num)]
        self.cross_bias = [
            self.add_weight(name='b_' + str(i),
                            shape=(dim, 1),
                            initializer='random_normal',
                            regularizer=l2(self.reg_b),
                            trainable=True
                            )
            for i in range(self.layer_num)]

    def call(self, inputs):
        x_0 = tf.expand_dims(inputs, axis=2)  # (batch_size, dim, 1)
        x_l = x_0  # (None, dim, 1)
        for i in range(self.layer_num):
            x_l1 = tf.tensordot(x_l, self.cross_weights[i], axes=[1, 0])  # (batch_size, 1, 1)
            print(x_l1)
            x_l = tf.matmul(x_0, x_l1) + self.cross_bias[i] + x_l  # (batch_size, dim, 1)
        x_l = tf.squeeze(x_l, axis=2)  # (batch_size, dim)
        return x_l


class New_FM(Layer):
    def __init__(self, feature_length, w_reg=1e-6):
        """New Factorization Machine for DeepFm.
        Note. In DeepFM, only the first order feature and second order feature intersect are included.
        Args:
            :param feature_length: A scalar. The length of features.
            :param w_reg: A scalar. The regularization coefficient of parameter w.
        :return:
        """
        super(New_FM, self).__init__()
        self.feature_length = feature_length
        self.w_reg = w_reg

    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=(self.feature_length, 1),
                                 initializer='random_normal',
                                 regularizer=l2(self.w_reg),
                                 trainable=True)

    def call(self, inputs, **kwargs):
        """
        :param inputs: A dict with shape `(batch_size, {'sparse_inputs', 'embed_inputs'})`:
          sparse_inputs is 2D tensor with shape `(batch_size, sum(field_num))`
          embed_inputs is 3D tensor with shape `(batch_size, fields, embed_dim)`
        """
        sparse_inputs, embed_inputs = inputs['sparse_inputs'], inputs['embed_inputs']
        sparse_inputs = tf.concat([value for _, value in sparse_inputs.items()], axis=-1)
        # first order
        first_order = tf.reduce_sum(tf.nn.embedding_lookup(self.w, sparse_inputs), axis=1)  # (batch_size, 1)
        # second order
        square_sum = tf.square(tf.reduce_sum(embed_inputs, axis=1, keepdims=True))  # (batch_size, 1, embed_dim)
        sum_square = tf.reduce_sum(tf.square(embed_inputs), axis=1, keepdims=True)  # (batch_size, 1, embed_dim)
        second_order = 0.5 * tf.reduce_sum(square_sum - sum_square, axis=2)  # (batch_size, 1)
        return first_order + second_order


class CIN(Layer):
    def __init__(self, cin_size, l2_reg=0.):
        """CIN
        Args:
            :param cin_size: A list. [H_1, H_2 ,..., H_k], a list of the number of layers
            :param l2_reg: A scalar. L2 regularization.
        :return:
        """
        super(CIN, self).__init__()
        self.cin_size = cin_size
        self.l2_reg = l2_reg

    def build(self, input_shape):
        # get the number of embedding fields
        self.embedding_nums = input_shape[1]
        # a list of the number of CIN
        self.field_nums = [self.embedding_nums] + self.cin_size
        # filters
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),
                initializer='random_normal',
                regularizer=l2(self.l2_reg),
                trainable=True)
            for i in range(len(self.field_nums) - 1)
        }

    def call(self, inputs):
        dim = inputs.shape[-1]
        hidden_layers_results = [inputs]
        # split dimension 2 for convenient calculation
        split_X_0 = tf.split(hidden_layers_results[0], dim, 2)  # dim * (None, field_nums[0], 1)
        for idx, size in enumerate(self.cin_size):
            split_X_K = tf.split(hidden_layers_results[-1], dim, 2)  # dim * (None, filed_nums[i], 1)
            result_1 = tf.matmul(split_X_0, split_X_K, transpose_b=True)  # (dim, None, field_nums[0], field_nums[i])
            result_2 = tf.reshape(result_1, shape=[dim, -1, self.embedding_nums * self.field_nums[idx]])
            result_3 = tf.transpose(result_2, perm=[1, 0, 2])  # (None, dim, field_nums[0] * field_nums[i])
            result_4 = tf.nn.conv1d(input=result_3, filters=self.cin_W['CIN_W_' + str(idx)], stride=1,
                                    padding='VALID')
            result_5 = tf.transpose(result_4, perm=[0, 2, 1])  # (None, field_num[i+1], dim)
            hidden_layers_results.append(result_5)
        final_results = hidden_layers_results[1:]
        result = tf.concat(final_results, axis=1)  # (None, H_1 + ... + H_K, dim)
        result = tf.reduce_sum(result,  axis=-1)  # (None, dim)
        return result


class LBA(Layer):
    def __init__(self, dnn_dropout=0.):
        """location-based attention layer. FISSA
        Args:
            dnn_dropout: A scalar.
        :return:
        """
        super(LBA, self).__init__()
        self.dropout = Dropout(dnn_dropout)

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.W = self.add_weight(
            shape=[self.dim, self.dim],
            name='att_weights',
            initializer='random_normal')
        self.W_2 = self.add_weight(
            shape=[self.dim, self.dim],
            name='att_weights2',
            initializer='random_normal')
        self.q = self.add_weight(
            shape=[1, self.dim],
            name='query weight',
            initializer='random_normal')

    def call(self, inputs):
        k, v, mask = inputs  # (None, seq_len, dim), (None, seq_len, dim), (None, seq_len, 1)
        k = tf.matmul(k, self.W)  # (None, seq_len, dim)
        mat_qk = tf.matmul(k, self.q, transpose_b=True)  # (None, seq_len, 1)
        dk = tf.cast(self.dim, dtype=tf.float32)
        # Scaled
        scaled_att_logits = mat_qk / tf.sqrt(dk)
        # Mask
        paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask, 0), paddings, scaled_att_logits)  # (None, seq_len, 1)
        # softmax
        outputs = tf.nn.softmax(logits=tf.transpose(outputs, [0, 2, 1]), axis=-1)  # (None, 1, seq_len)
        # output
        outputs = tf.squeeze(tf.matmul(tf.matmul(outputs, v), self.W_2), axis=1)  # (None, dim)
        outputs = self.dropout(outputs)
        return outputs


class FFN(Layer):
    def __init__(self, hidden_unit, d_model):
        """Feed Forward Network.
        Args:
            :param hidden_unit: A scalar.
            :param d_model: A scalar.
        :return:
        """
        super(FFN, self).__init__()
        self.conv1 = Conv1D(filters=hidden_unit, kernel_size=1, activation='relu', use_bias=True)
        self.conv2 = Conv1D(filters=d_model, kernel_size=1, activation=None, use_bias=True)

    def call(self, inputs):
        x = self.conv1(inputs)
        output = self.conv2(x)
        return output


class TransformerEncoder(Layer):
    def __init__(self, d_model, num_heads=1, ffn_hidden_unit=128, dropout=0., layer_norm_eps=1e-6):
        """Encoder Layer.
        Args:
            :param d_model: A scalar. The self-attention hidden size.
            :param num_heads: A scalar. Number of heads.
            :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
            :param dropout: A scalar. Number of dropout.
            :param layer_norm_eps: A scalar. Small float added to variance to avoid dividing by zero.
        :return:
        """
        super(TransformerEncoder, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FFN(ffn_hidden_unit, d_model)

        self.layernorm1 = LayerNormalization(epsilon=layer_norm_eps)
        self.layernorm2 = LayerNormalization(epsilon=layer_norm_eps)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs):
        x, mask = inputs
        # self-attention
        att_out = self.mha(x, x, x, mask)  # (None, seq_len, d_model)
        att_out = self.dropout1(att_out)
        # residual add
        out1 = self.layernorm1(x + att_out)  # (None, seq_len, d_model)
        # ffn
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        # residual add
        out2 = self.layernorm2(out1 + ffn_out)  # (None, seq_len, d_model)
        return out2