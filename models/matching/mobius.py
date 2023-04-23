import tensorflow as tf
from backend.utils.preprocess_utils import get_preprocess_layers
from backend.blocks.mlp import create_mlp
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense, BatchNormalization
from config_parser.configuration import Configuration


class Mobius(Model):
    """
    莫比乌斯流程实现
    """
    def __init__(self, feature_conf: Configuration, loss, judge_evaluator: Model, name=None):
        super(Mobius, self).__init__(name=name)
        self.feature_conf = feature_conf
        self.preprocessor = get_preprocess_layers(feature_conf)
        self.loss = loss

        self.query_tower = create_mlp([1024, 768, 512, 256], 0.5, "relu", BatchNormalization(epsilon=1e-6), name="query_tower")
        self.ad_tower = create_mlp([1024, 768, 512, 256], 0.5, "relu", BatchNormalization(epsilon=1e-6), name="ad_tower")

        self.query_features = self.feature_conf.features.get_features(tower="user")
        self.ad_features = self.feature_conf.features.get_features(tower="ad")

        self.rel_judge_query_feature = self.feature_conf.features.get_features(tower="user")
        self.rel_judge_ad_feature = self.feature_conf.features.get_features(tower="ad")
        self.rel_judge_evaluator = judge_evaluator

    @staticmethod
    def make_cross_data(query_embedding, doc_embedding):
        data_nums = query_embedding.shape[0]
        tf.ones([data_nums, data_nums])

    def call(self):
        pass

