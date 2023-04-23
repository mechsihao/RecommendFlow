import tensorflow as tf

from backend.utils.preprocess_utils import get_preprocess_layers
from backend.blocks.mlp import create_mlp
from backend.blocks.bert import get_multi_input_bert_model
from backend.layers.fusion_layers import AttentionFusion
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, BatchNormalization
from config_parser.configuration import Configuration


class Que2Search(Model):
    def __init__(self, feature_conf: Configuration, loss, aux_alpha=0, name=None):
        super(Que2Search, self).__init__(name=name)
        self.feature_conf = feature_conf
        self.preprocessor = get_preprocess_layers(feature_conf)
        self.loss = loss
        self.aux_alpha = aux_alpha
        self.query_cols = feature_conf.features.get_fields(tower="user")
        self.app_cols = feature_conf.features.get_fields(tower="ad")
        self.bert_fields = feature_conf.features.get_fields(tower="user", deal="token_id")
        self.clk_label, self.app_label = feature_conf.features.labels
        self.app_label_cat_nums = feature_conf.features.labels[1].vocab_size

        self.embedding_dim = self.bert_conf['dim']

        self.bert_conf = feature_conf.get_conf_value("bert_conf")

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

        self.embedding_dim = feature_conf.get_conf_value("dim")
        self.query_attention_fusion = AttentionFusion(self.embedding_dim, len(self.query_cols), name="query_fusion", is_norm=False)
        self.app_attention_fusion = AttentionFusion(self.embedding_dim, len(self.app_cols), name="app_fusion", is_norm=True)

        # 这部分是否需要还需要实验
        self.query_embedding_dense = Dense(self.embedding_dim, activation="linear", name="query_embedding_dense")
        self.app_embedding_dense = Dense(self.embedding_dim, activation="linear", name="app_embedding_dense")

        units = [self.embedding_dim]
        activation = "selu"
        norm = BatchNormalization
        dropout_rate = 0.3

        self.query_dense = create_mlp(units, dropout_rate, activation, norm(epsilon=1e-6), name="query_mlp")
        self.query_sug_dense = create_mlp(units, dropout_rate, activation, norm(epsilon=1e-6), name="query_sug_mlp")
        self.query_2gram_dense = create_mlp(units, dropout_rate, activation, norm(epsilon=1e-6), name="query_2gram_mlp")

        self.app_name_dense = create_mlp(units, dropout_rate, activation, norm(epsilon=1e-6), name="app_name_mlp")
        self.app_desc_dense = create_mlp(units, dropout_rate, activation, norm(epsilon=1e-6), name="app_desc_mlp")
        self.app_id_dense = create_mlp(units, dropout_rate, activation, norm(epsilon=1e-6), name="app_id_mlp")
        self.app_kws_dense = create_mlp(units, dropout_rate, activation, norm(epsilon=1e-6), name="app_kws_mlp")
        self.app_name_2gram_dense = create_mlp(units, dropout_rate, activation, norm(epsilon=1e-6), name="app_name_2gram_mlp")
        self.app_desc_2gram_dense = create_mlp(units, dropout_rate, activation, norm(epsilon=1e-6), name="app_desc_2gram_mlp")

        # self.app_classify_dense = create_mlp([self.app_label_cat_nums], dropout_rate, "softmax", norm(epsilon=1e-6), name="app_cls")

    @tf.function
    def get_query_embedding(self, q):
        query_2gram = self.preprocessor["query_2gram"](q["query_2gram"])
        q_embeddings = [
            self.query_2gram_dense(query_2gram)
        ]
        return q_embeddings

    @tf.function
    def get_app_embedding(self, d):
        app_id = self.preprocessor["app_id"](d["app_id"])
        app_kws = self.preprocessor["app_kws"](d["app_kws"])
        app_name_2gram = self.preprocessor["app_name_2gram"](d["app_name_2gram"])
        app_desc_2gram = self.preprocessor["app_desc_2gram"](d["app_desc_2gram"])
        d_embeddings = [
            self.app_id_dense(app_id),
            self.app_kws_dense(app_kws),
            self.app_name_2gram_dense(app_name_2gram),
            self.app_desc_2gram_dense(app_desc_2gram)
        ]
        return d_embeddings

    @tf.function
    def get_query_embedding_v2(self, q):
        return tf.nn.l2_normalize(self.query_embedding_dense(q), 1)

    @tf.function
    def get_app_embedding_v2(self, d):
        return tf.nn.l2_normalize(self.app_embedding_dense(d), 1)

    @tf.function
    def get_query_bert_embedding(self, train_data):
        bert_features = [[train_data[name + "_tok_id"], train_data[name + "_seg_id"]] for name in self.bert_cols[:2]]
        query_bert, query_sug_bert = self.query_bert_encoder(bert_features)
        return [
            self.query_dense(query_bert),
            self.query_sug_dense(query_sug_bert)
        ]

    @tf.function
    def get_app_bert_embedding(self, train_data):
        bert_features = [[train_data[name + "_tok_id"], train_data[name + "_seg_id"]] for name in self.bert_cols[2:]]
        app_name_bert, app_desc_bert = self.app_bert_encoder(bert_features)
        return [
            self.app_name_dense(app_name_bert),
            self.app_desc_dense(app_desc_bert)
        ]

    def call(self, train_data, training=False, mask=None):
        query_features = {name: train_data[name] for name in self.query_cols if name not in self.bert_cols}
        app_features = {name: train_data[name] for name in self.app_cols if name not in self.bert_cols}

        q_embedding = self.get_query_embedding(query_features)
        a_embedding = self.get_app_embedding(app_features)

        q_bert = self.get_query_bert_embedding(train_data)
        a_bert = self.get_app_bert_embedding(train_data)

        q_embedding = self.query_attention_fusion(q_bert + q_embedding, training)
        a_embedding = self.app_attention_fusion(a_bert + a_embedding, training)

        q_embedding = self.query_embedding_dense(q_embedding)
        a_embedding = self.app_embedding_dense(a_embedding)

        y_true = tf.squeeze(train_data[self.clk_label], -1)
        #
        # # 多任务辅助loss，增强app侧理解
        # a_true = tf.squeeze(train_data[self.app_label_col], -1)
        # a_pred = self.app_classify_dense(a_embedding)

        if training:
            y_pred = tf.reduce_sum(q_embedding * a_embedding, axis=1, keepdims=True, name="cos")
            main_loss = self.loss(y_true, q_embedding, a_embedding)
            auc_loss = 0  # args.aux_loss(a_true, a_pred)
            self.add_loss((1-self.aux_alpha) * main_loss + self.aux_alpha * auc_loss)
            return {self.clk_label: y_pred}  # {self.clk_label_col: y_pred, self.app_label_col: a_pred}
        else:
            return {
                "query": q_embedding,
                "app": a_embedding,
                "label": y_true,
                "app_id": tf.squeeze(train_data["app_id"], -1),
                "down": tf.squeeze(train_data["down"], -1),
                # "app_label": a_true
            }

    def get_fusion_weights(self):
        res = {
            "query": self.query_attention_fusion.get_fusion_weights(),
            "app": self.app_attention_fusion.get_fusion_weights()
        }
        return res

    def get_weights_all_nums(self):
        pass

    def get_config(self):
        config = super(Que2Search, self).get_config()
        return config
