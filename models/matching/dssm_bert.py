import tensorflow as tf

from backend.utils.preprocess_utils import get_preprocess_layers
from backend.blocks.bert import get_multi_input_bert_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense
import tensorflow.keras.backend as K
from config_parser.configuration import Configuration


class BertModel(Model):

    def __init__(self, feature_conf: Configuration, loss=None, name=None):
        super(BertModel, self).__init__(name=name)
        self.feature_conf = feature_conf
        self.preprocessor = get_preprocess_layers(feature_conf)

        self.label_col = feature_conf.features.labels[0].name

        self.query_bert_conf = feature_conf.get_conf_value("user_bert_conf")
        self.app_bert_conf = feature_conf.get_conf_value("ad_bert_conf")

        self.query_bert_cols = feature_conf.features.get_features(tower="user")
        self.app_bert_cols = feature_conf.features.get_features(tower="ad")

        self.query_bert = get_multi_input_bert_model(
            feature_list=self.query_bert_cols,
            conf_path=self.query_bert_conf['conf_path'],
            ckpt_path=self.query_bert_conf['ckpt_path'],
            is_trainable=self.query_bert_conf['finetune'],
            pooling=self.query_bert_conf['pooling'],
            out_layer=self.query_bert_conf['out_layer'],
            maxlen=self.query_bert_conf['max_len'],
            name='q_bert_encoder'
        )

        self.app_bert = get_multi_input_bert_model(
            feature_list=self.app_bert_cols,
            conf_path=self.app_bert_conf['conf_path'],
            ckpt_path=self.app_bert_conf['ckpt_path'],
            is_trainable=self.app_bert_conf['finetune'],
            pooling=self.app_bert_conf['pooling'],
            out_layer=self.app_bert_conf['out_layer'],
            maxlen=self.app_bert_conf['max_len'],
            name='a_bert_encoder'
        )

        self.embedding_dim = feature_conf['tower_conf']['embedding_dim']

        self.query_embedding_dense = Dense(self.embedding_dim, activation="linear", name="query_embedding_dense")
        self.app_embedding_dense = Dense(self.embedding_dim, activation="linear", name="app_embedding_dense")

        self.loss = loss or feature_conf

        # self.query_attention_fusion = AttentionFusion(self.embedding_dim, len(self.query_bert_cols), name="query_fusion", is_norm=False)
        # self.app_attention_fusion = AttentionFusion(self.embedding_dim, len(self.app_bert_cols), name="app_fusion", is_norm=True)

    @tf.function
    def get_query_bert_embedding(self, bert_features):
        return self.query_bert(bert_features)

    @tf.function
    def get_app_bert_embedding(self, bert_features):
        return self.app_bert(bert_features)

    @tf.function
    def query_fusion(self, embedding_list):
        return self.query_attention_fusion(embedding_list)

    @tf.function
    def app_fusion(self, embedding_list):
        return self.app_attention_fusion(embedding_list)

    def call(self, train_data, training=False, mask=None):
        q_bert_features = [[train_data[name + "_tok_id"], train_data[name + "_seg_id"]] for name in self.query_bert_cols]
        a_bert_features = [[train_data[name + "_tok_id"], train_data[name + "_seg_id"]] for name in self.app_bert_cols]

        q_embedding_list = self.get_query_bert_embedding(q_bert_features)
        a_embedding_list = self.get_app_bert_embedding(a_bert_features)

        q_embedding_list = [q_embedding_list] if len(self.query_bert_cols) == 1 else q_embedding_list
        a_embedding_list = [a_embedding_list] if len(self.app_bert_cols) == 1 else a_embedding_list

        q_embedding = K.concatenate(q_embedding_list)
        a_embedding = K.concatenate(a_embedding_list)

        q_embedding = self.query_embedding_dense(q_embedding)
        a_embedding = self.app_embedding_dense(a_embedding)

        q_embedding = K.l2_normalize(q_embedding, axis=1)
        a_embedding = K.l2_normalize(a_embedding, axis=1)

        y_true = tf.squeeze(train_data[self.label_col], -1)

        if training:
            main_loss = self.loss(y_true, q_embedding, a_embedding)
            self.add_loss(main_loss)
        else:
            return {
                "query": q_embedding,
                "app": a_embedding,
                "label": y_true,
                "app_id": tf.squeeze(train_data["app_id"], -1),
                "down": tf.squeeze(train_data["down"], -1)
            }

    def get_config(self):
        config = super(BertModel, self).get_config()
        return config
