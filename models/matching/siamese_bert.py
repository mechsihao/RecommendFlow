import tensorflow as tf
from tensorflow.python.keras.layers import Dense

from backend.layers.fusion_layers import AttentionFusion
from backend.utils.preprocess_utils import get_preprocess_layers
from backend.blocks.bert import get_multi_input_bert_model
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from config_parser.configuration import Configuration


class BertModel(Model):

    def __init__(self, feature_conf: Configuration, loss, name=None):
        super(BertModel, self).__init__(name=name)
        self.feature_conf = feature_conf
        self.preprocessor = get_preprocess_layers(feature_conf)
        self.loss_fun = loss

        self.label_col = feature_conf.features.label_names[0]
        self.embedding_dim = feature_conf.networks["embedding_dim"]
        self.embedding_pooling = feature_conf.networks["embedding_pooling"]

        self.bert_conf = feature_conf.get_conf_value("bert_conf")
        self.bert_dim = feature_conf.networks["bert_conf"]["dim"]
        self.bert_fields = feature_conf.features.get_fields(deal="token_id")
        self.bert_fields_map = feature_conf.features.get_fields_map(deal="token_id", name_only=True)
        self.bert_fields_group = feature_conf.features.get_fields_feature_tuple(deal="token_id", name_only=True)

        self.query_fields = feature_conf.features.get_fields(deal="token_id", tower="user")
        self.query_fields_index = feature_conf.features.index_of_fields(self.query_fields, deal="token_id")

        self.app_fields = feature_conf.features.get_fields(deal="token_id", tower="ad")
        self.app_fields_index = feature_conf.features.index_of_fields(self.app_fields, deal="token_id")

        self.bert = get_multi_input_bert_model(
            feature_list=self.bert_fields,
            conf_path=self.bert_conf['conf_path'],
            ckpt_path=self.bert_conf['ckpt_path'],
            is_trainable=self.bert_conf['finetune'],
            pooling=self.bert_conf['pooling'],
            out_layer=self.bert_conf['out_layer'],
            maxlen=self.bert_conf['max_len'],
            name='bert_encoder'
        )

        if self.embedding_pooling == "dense":
            self.query_dense = Dense(self.embedding_dim, activation="linear", name="query_dense")
            self.app_dense = Dense(self.embedding_dim, activation="linear", name="app_dense")
        elif self.embedding_pooling == "attention":
            self.query_dense = AttentionFusion(self.bert_dim, len(self.query_fields), name="query_attention")
            self.app_dense = AttentionFusion(self.bert_dim, len(self.app_fields), name="app_attention")

        self.auc = tf.keras.metrics.AUC()

    @tf.function
    def get_bert_embedding(self, bert_features):
        bert_out = self.bert(bert_features)
        query_bert = [bert_out[i] for i in self.query_fields_index]
        app_bert = [bert_out[i] for i in self.app_fields_index]
        return query_bert, app_bert

    @tf.function
    def query_pooling_embedding(self, embedding_list):
        if self.embedding_pooling in ("dense", "attention"):
            return self.query_dense(K.concatenate(embedding_list, axis=1))
        else:
            return tf.reduce_sum(embedding_list, axis=0) if self.embedding_pooling == "sum" else tf.reduce_mean(embedding_list, axis=0)

    @tf.function
    def app_pooling_embedding(self, embedding_list):
        if self.embedding_pooling in ("dense", "attention"):
            return self.app_dense(K.concatenate(embedding_list, axis=1))
        else:
            return tf.reduce_sum(embedding_list, axis=0) if self.embedding_pooling == "sum" else tf.reduce_mean(embedding_list, axis=0)

    def call(self, train_data, training=False, mask=None):
        bert_features = [[train_data[feature] for feature in group] for group in self.bert_fields_group]

        q_embedding_list, a_embedding_list = self.get_bert_embedding(bert_features)

        q_embedding = self.query_pooling_embedding(q_embedding_list)
        a_embedding = self.app_pooling_embedding(a_embedding_list)

        q_embedding = K.l2_normalize(q_embedding, axis=1)
        a_embedding = K.l2_normalize(a_embedding, axis=1)

        y_pred = tf.reduce_sum(q_embedding * a_embedding, axis=1)
        y_true = tf.squeeze(train_data[self.label_col], -1)

        self.add_loss(self.loss_fun(y_true, q_embedding, a_embedding))
        self.add_metric(self.auc(y_true, y_pred), name="auc")

        if training:
            return {self.label_col: y_true}
        else:
            return {
                self.label_col: y_true,
                "y_true": y_true,
                "y_pred": y_pred,
                "query": q_embedding,
                "app": a_embedding,
                "app_id": tf.squeeze(train_data["app_id"], -1),
                "down": tf.squeeze(train_data["down"], -1)
            }

    def get_config(self):
        config = super(BertModel, self).get_config()
        config.update({"embedding_dim": self.embedding_dim})
        config.update({"embedding_pooling": self.embedding_pooling})
        return config
