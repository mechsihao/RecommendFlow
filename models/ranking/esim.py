import tensorflow as tf
import tensorflow.keras.backend as K

from backend.blocks.mlp import create_mlp
from tensorflow.python.keras.models import Model
from config_parser.configuration import Configuration
from backend.layers.attention_layers import SoftAttention
from backend.blocks.bert import get_multi_input_bert_model
from backend.utils.preprocess_utils import get_preprocess_layers
from tensorflow.python.keras.layers import Dense, Lambda, LayerNormalization, GlobalAveragePooling1D, GlobalMaxPooling1D, Dropout


class Esim(Model):
    """Esim模型利用soft-attention，对双塔query做交叉attention，融合了dense特征信息，做ctr预估
    """
    def __init__(self, feature_conf: Configuration, loss, name=None):
        super(Esim, self).__init__(name=name)
        self.feature_conf = feature_conf
        self.preprocessor = get_preprocess_layers(feature_conf)
        self.loss_fun = loss

        self.label_col = feature_conf.features.label_names[0]
        self.embedding_dim = feature_conf.networks["embedding_dim"]
        self.embedding_pooling = feature_conf.networks["embedding_pooling"]

        self.bert_conf = feature_conf.get_conf_value("bert_conf")
        self.bert_dim = feature_conf.networks["bert_conf"]["dim"]
        self.bert_fields = feature_conf.features.feature_except(deal="token_id")

        self.dense_fields = feature_conf.features.get_fields_except(deal="token_id")
        self.query_token_fields = feature_conf.features.get_fields(deal="token_id", tower="user")
        self.app_token_fields = feature_conf.features.get_fields(deal="token_id", tower="ad")

        self.bert = get_multi_input_bert_model(
            feature_list=self.bert_fields,
            conf_path=self.bert_conf['conf_path'],
            ckpt_path=self.bert_conf['ckpt_path'],
            is_trainable=self.bert_conf['finetune'],
            pooling=None,
            out_layer=self.bert_conf['out_layer'],
            maxlen=self.bert_conf['max_len'],
            name='bert_encoder'
        )

        self.input_mlp = create_mlp(
            [(len(self.dense_fields) - len(self.query_token_fields) - len(self.app_token_fields)) * 128, 512],
            0.3, tf.keras.activations.gelu, LayerNormalization(epsilon=1e-6)
        )
        self.soft_attention = SoftAttention()
        self.sub = Lambda(lambda x: x[0] - x[1])
        self.mul = Lambda(lambda x: x[0] * x[1])
        self.output_mlp = create_mlp([1024, 512], 0.3, tf.keras.activations.gelu, LayerNormalization(epsilon=1e-6))
        self.dense_output = Dense(units=2, activation='softmax', name="ctr")

        self.drop_out = Dropout(rate=0.3)
        self.avg_pooling = GlobalAveragePooling1D()
        self.max_pooling = GlobalMaxPooling1D()

    @tf.function
    def get_query_bert_embedding(self, bert_features):
        q_embedding_list = self.query_bert(bert_features)
        return [q_embedding_list] if len(self.query_bert_cols) == 1 else q_embedding_list

    @tf.function
    def get_app_bert_embedding(self, bert_features):
        a_embedding_list = self.app_bert(bert_features)
        return [a_embedding_list] if len(self.app_bert_cols) == 1 else a_embedding_list

    def call(self, train_data, training=False, mask=None):
        dense_features = [train_data[name] for name in self.dense_fields]
        q_bert_features = [[train_data[name + "_tok_id"], train_data[name + "_seg_id"]] for name in self.query_bert_cols]
        a_bert_features = [[train_data[name + "_tok_id"], train_data[name + "_seg_id"]] for name in self.app_bert_cols]

        dense_features = K.concatenate(dense_features)
        d_embedding = self.input_mlp(dense_features)

        q, a = K.concatenate(self.get_query_bert_embedding(q_bert_features)), K.concatenate(self.get_app_bert_embedding(a_bert_features))
        att_q, att_a = self.soft_attention([q, a])
        sub_q_att, sub_a_att = self.sub([q, att_q]), self.sub([a, att_a])
        mul_q_att, mul_a_att = self.mul([q, att_q]), self.mul([a, att_a])
        m_q, m_a = K.concatenate([q, att_q, sub_q_att, mul_q_att], axis=1), K.concatenate([a, att_a, sub_a_att, mul_a_att], axis=1)
        avg_q, max_q, avg_a, max_a = self.avg_pooling(m_q), self.max_pooling(m_q), self.avg_pooling(m_a), self.max_pooling(m_a)

        pooled = K.concatenate([d_embedding, avg_q, max_q, avg_a, max_a, avg_q - avg_a, max_q - max_a], axis=1)
        x = self.drop_out(pooled)
        x = self.output_mlp(x)

        y_pred = self.dense_output(x)
        return {self.label_col: y_pred}

    def get_config(self):
        config = super(Esim, self).get_config()
        return config
