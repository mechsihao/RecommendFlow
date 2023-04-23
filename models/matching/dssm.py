import tensorflow as tf
import tensorflow.keras.backend as K

from backend.blocks.mlp import create_mlp
from tensorflow.python.keras.models import Model
from config_parser.configuration import Configuration
from tensorflow.python.keras.layers import BatchNormalization
from backend.utils.preprocess_utils import get_preprocess_layers


class Dssm(Model):

    def __init__(self, feature_conf: Configuration, loss=None, name=None):
        super(Dssm, self).__init__(name=name)
        self.feature_conf = feature_conf
        self.preprocessor = get_preprocess_layers(feature_conf)

        self.label_col = feature_conf.features.labels[0]
        self.imei_col = feature_conf.features.labels[1]
        self.ad_col = feature_conf.features.labels[2]

        self.user_cols = feature_conf.features.get_features(tower="user")
        self.ad_cols = feature_conf.features.get_features(tower="ad")

        self.user_dense = create_mlp([1024, 512, 256], 0.3, "selu", BatchNormalization(epsilon=1e-6), name='user_dense_tower')
        self.ad_dense = create_mlp([1024, 512, 256], 0.3, "selu", BatchNormalization(epsilon=1e-6), name='ad_dense_tower')

        self.loss_fun = loss

    @tf.function
    def embedding_concat(self, embedding_list):
        return K.concatenate(embedding_list, axis=-1)

    @tf.function
    def embedding_norm(self, embedding):
        return K.l2_normalize(embedding)

    def call(self, train_data, training=False, mask=None):
        user_features = [[train_data[name + "_tok_id"], train_data[name + "_seg_id"]] for name in self.user_cols]
        ad_features = [[train_data[name + "_tok_id"], train_data[name + "_seg_id"]] for name in self.ad_cols]

        u_embedding = self.embedding_concat(user_features)
        a_embedding = self.embedding_concat(ad_features)

        u_embedding = self.embedding_norm(u_embedding)
        a_embedding = self.embedding_norm(a_embedding)

        y_true = tf.squeeze(train_data[self.label_col], -1)

        if training:
            main_loss = self.loss(y_true, u_embedding, a_embedding)
            self.add_loss(main_loss)
        else:
            return {
                "user": u_embedding,
                "ad": a_embedding,
                "label": y_true,
                "ad_id": tf.squeeze(train_data[self.ad_col], -1),
                "imei": tf.squeeze(train_data[self.imei_col], -1)
            }

    def get_config(self):
        config = super(Dssm, self).get_config()
        return config
