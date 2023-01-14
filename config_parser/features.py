"""
创建人：MECH
创建时间：2021/12/19
功能描述：负责配置文件中的特征解析
"""

import os
from typing import Dict, List, Union, Any

import tensorflow as tf

from case_class.case_class import CaseClass
from config_parser.config_proto import FeatureTower, FeatureDeal, FeaturePooling
from utils.hdfs_util import ls_hdfs_paths
from utils.str_parser import str2list
from utils.util import read_csv
from config_parser.config_utils import load_vocab


TYPE_INT = "int"
TYPE_FLOAT = "float"
TYPE_STR = "str"

SUPPORT_TYPE = [TYPE_INT, TYPE_FLOAT, TYPE_STR]
TYPE_MAP = {TYPE_INT: tf.int32, TYPE_FLOAT: tf.float32, TYPE_STR: tf.string}
DEFAULT_MAP = {TYPE_INT: 0, TYPE_FLOAT: 0.0, TYPE_STR: ""}


class Feature(CaseClass):
    def __init__(self,
                 name: str,
                 field_name: str,
                 ftype: str,
                 tower: FeatureTower,
                 deal: FeatureDeal,
                 vocab_size: int = -1,
                 embedding_dim: int = None,
                 pooling: FeaturePooling = FeaturePooling("null"),
                 working: bool = True,
                 vocabs: Union[List[str], str] = None,
                 seeds: Union[List[int], int] = None):
        self.name = name
        self.field_name = field_name
        assert ftype.lower() in SUPPORT_TYPE, f"Feature type field only support: {SUPPORT_TYPE}, got {ftype}, field: {field_name}"
        self.type = TYPE_MAP[ftype.lower()]
        self.tower = tower
        self.deal = deal
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.default = DEFAULT_MAP[ftype.lower()]
        self.working: bool = working
        self.vocabs = [eval(ftype.lower())(vocab) for vocab in vocabs] if isinstance(vocabs, list) else vocabs
        self.hash_seeds = seeds

    def is_auto_vocabs(self):
        return self.vocabs == "__AUTO__"

    def is_token_id(self):
        return self.deal == FeatureDeal.TokenId

    def is_lookup(self):
        return self.deal == FeatureDeal.Lookup

    def is_string_lookup(self):
        return self.deal == FeatureDeal.StringLookup

    def is_int_lookup(self):
        return self.deal == FeatureDeal.IntegerLookup

    def is_hashing(self):
        return self.deal == FeatureDeal.Hashing

    def is_discrete(self):
        return self.deal == FeatureDeal.Discrete

    def is_image(self):
        return self.deal == FeatureDeal.Image

    def is_embedding(self):
        return self.deal == FeatureDeal.Embedding

    def is_numeric(self):
        return self.deal == FeatureDeal.Numeric

    def is_bert_encode(self):
        return self.deal == FeatureDeal.BertEncode

    def __hash__(self):
        return hash(self.name)


class Features(object):
    """
    配置特征示例：
        - feature_field: name, field, type, tower, deal, vocab, embedding_dim, pooling, working
        - name: 特征名称
        - type: 特征类型，只能是int、float、str三种，都是多值特征
        - tower: 塔标识，对于多塔模型，该值只有
    """

    def __init__(self, conf, vocabs_map: Dict[str, Any] = None, seeds: Union[int, List[int]] = None):
        self.conf = conf

        feature_fields = conf['Features']['feature_fields']
        self.field_names = feature_fields if isinstance(feature_fields, list) else str2list(feature_fields)
        self.vocabs_map = vocabs_map or {}
        self.seeds = seeds

        self.feature_group = self.__init_feature_group(conf['Features']['feature_group'] if 'feature_group' in conf['Features'] else {})
        self.features = self.__init_features()

        self.__set_attr_by_deal_fun()

    @property
    def train_features(self):
        return [feature for feature in self.features if feature.working]

    @property
    def train_feature_names(self):
        return [feature.name for feature in self.features if feature.working]

    @property
    def user_features(self):
        return self.get_tower_features("user")

    @property
    def ad_features(self):
        return self.get_tower_features("ad")

    @property
    def context_features(self):
        return self.get_tower_features("context")

    @property
    def labels(self):
        return self.get_tower_features("label")

    def __check_feature_exist(self, name: str):
        assert self.contains(name), f"Feature={name} dose not exists."

    def __check_feature_field_exist(self, field: str):
        assert self.contains_field(field), f"Feature field={field} dose not exists."

    def __init_features(self) -> List[Feature]:
        res, res_name = [], {}
        for conf in self.conf["Features"]["features"]:
            feature_list = self.__parse_feature(conf)
            for feature in feature_list:
                if feature.name in res_name:
                    raise Exception(f"Feature: [{self.field_names[0]}='{feature.field_name}', name='{feature.name}'] was conflicted with "
                                    f"Feature: [{self.field_names[0]}='{res_name[feature.name]}', name='{feature.name}']")
                else:
                    res_name[feature.name] = feature.field_name
            res.extend(feature_list)

        return res

    @staticmethod
    def __init_feature_group(org_group) -> Dict[str, List[str]]:
        ret = {}
        for k, v in org_group.items():
            if isinstance(v, str):
                ret[k.lower()] = str2list(v)
            elif isinstance(v, list):
                ret[k.lower()] = v
            else:
                raise Exception(f"Feature group except str or list, but got {type(v).__name__}.")
        return ret

    def __init_vocab(self) -> Dict[str, Union[str, List[str]]]:
        """vocab导入的时候先按照str类型来写入，到时候转化成特征相应的类型
        """
        return self.conf["Features"]["vocabs"] if "vocabs" in self.conf["Features"] else {}

    def __get_vocab(self, vocab_name, read: bool = True):
        vocab = self.vocabs_map[vocab_name]
        if isinstance(vocab, list):
            return vocab
        elif isinstance(vocab, str):
            if read:
                path = ls_hdfs_paths(vocab)[0] if vocab.startswith("hdfs://") else vocab
                vocab = read_csv(path, sep="\t", cache_data=True, columns=["vocab_id", "vocab_name"]).astype(str)
                vocab = vocab[vocab.columns[0]].unique().tolist()
                self.vocabs_map[vocab_name] = vocab
                return vocab
            else:
                return vocab
        else:
            raise Exception(f"Vocab={vocab_name}, value={vocab}, type={type(vocab)}, expect list or string.")

    def __parse_feature(self, conf_str: str) -> List[Feature]:
        d = {f: v for f, v in zip(self.field_names, conf_str)}
        assert len(d) == len(self.field_names), f"Conf_str = {conf_str} is invalid, please check."

        field = d[self.field_names[0]].lower()
        name_list = self.feature_group[field] if field in self.feature_group else [field]
        ftype = d["type"].lower()
        tower = FeatureTower(d["tower"].lower())
        deal = FeatureDeal(d["deal"].lower())
        pooling = FeaturePooling(d["pooling"].lower())
        status = d["working"].lower() == "true"
        seeds = self.seeds if deal == FeatureDeal.Hashing else None
        vocab = d["vocab"].lower() if isinstance(d["vocab"], str) else d["vocab"]
        dim = -1 if deal in (
            FeatureDeal.Numeric, FeatureDeal.Null, FeatureDeal.TokenId, FeatureDeal.Image,
            FeatureDeal.Embedding, FeatureDeal.BertEncode
        ) else int(d["embedding_dim"])

        if deal in [FeatureDeal.StringLookup, FeatureDeal.IntegerLookup, FeatureDeal.Discrete] and status:
            # 这里故意设置为不论该特征是否有效都要校验vocab设置，防止配置文件乱配置
            if not isinstance(vocab, str):
                vocabs = vocab
                vocab_size = len(vocabs)
            else:
                if vocab.startswith("$"):
                    vocabs = self.__get_vocab(vocab[1:], read=True)
                    vocab_size = len(vocabs)
                else:
                    try:
                        vocab_size = int(vocab)
                        vocabs = "__AUTO__"
                        assert vocab_size > 0, "Vocab size must be set larger than 0, it means automatically adapt vocabs."
                    except ValueError as e:
                        if vocab == "null":
                            raise ValueError("Vocab or vocab size must be given in vocab field when "
                                             "feature deal method set in ['string_lookup', 'integer_lookup', 'discrete']")
                        elif vocab in self.vocabs_map:
                            raise Exception(f"Feature field: {field} get vocab symbol: '{vocab}', you may want to set as '${vocab}'?")
                        else:
                            raise Exception(f"Get unknown vocab symbol: '{vocab}', details: {str(e)}.")
        elif deal in [FeatureDeal.BertEncode]:
            vocabs = self.__get_vocab(vocab[1:], read=False) if vocab.startswith("$") else None
            if vocabs is None:
                raise Exception("Bert encode vocab must given.")
            elif os.path.isfile(vocabs):
                vocab_size = len(load_vocab(vocabs))
            else:
                raise FileNotFoundError(f"bert dict vocab path: {vocabs} dose not exist.")
        elif deal in [FeatureDeal.Hashing]:
            vocabs = None
            vocab_size = int(vocab)
        else:
            vocabs = None
            vocab_size = -1
        return [Feature(name, field, ftype, tower, deal, vocab_size, dim, pooling, status, vocabs, seeds) for name in name_list]

    def get_tower_features(self, tower: str, name_only: bool = False):
        return [feature.name if name_only else feature for feature in self.train_features if feature.tower == FeatureTower(tower)]

    def get_deal_features(self, deal: str, name_only: bool = False):
        return [feature.name if name_only else feature for feature in self.train_features if feature.deal == FeatureDeal(deal)]

    def get_fields_map(self, name_rlike: str = None, tower: str = None, deal: str = None, name_only: bool = False, train_only: bool = True):
        res: Dict[str, List[Union[Feature, str]]] = {}
        for feature in self.train_features if train_only else self.features:
            if filter_feature(feature, flag=name_rlike, tower=tower, deal=deal):
                if feature.field_name not in res:
                    res[feature.field_name] = [feature.name if name_only else feature]
                else:
                    res[feature.field_name].append(feature.name if name_only else feature)
        return res

    def get_fields(self, name_rlike: str = None, tower: str = None, deal: str = None, train_only: bool = True):
        return list(self.get_fields_map(name_rlike, tower, deal, True, train_only).keys())

    def index_of_fields(self, fields_list: List[str], name_rlike: str = None, tower: str = None, deal: str = None, train_only: bool = True):
        all_field_list = self.get_fields(name_rlike, tower, deal, train_only)
        return [all_field_list.index(i) for i in fields_list]

    def get_fields_feature_group(self, name_rlike: str = None, tower: str = None, deal: str = None, name_only=False, train_only=True):
        return list(self.get_fields_map(name_rlike, tower, deal, name_only, train_only).values())

    def get_feature(self, name: str):
        features = [feature for feature in self.train_features if feature.name == name]
        if len(features) == 0:
            raise Exception(f"Feature name = {name} dose not exist.")
        else:
            return features[0]

    def get_features(self, name_rlike: str = None, field: str = None, tower: str = None, deal: str = None, train_only: bool = True):
        return self.feature_filter(name_rlike, field, tower, deal, train_only)

    def index_of_features(self, names: List[str], name_rlike: str = None, field: str = None, tower: str = None, deal=None, train_only=True):
        all_features = [f.name for f in self.feature_filter(name_rlike, field, tower, deal, train_only)]
        return [all_features.index(name) for name in names]

    def feature_filter(self, name_rlike: str = None, field: str = None, tower: str = None, deal: str = None, train_only: bool = True):
        return [f for f in list(self.train_features if train_only else self.features) if filter_feature(f, name_rlike, field, tower, deal)]

    def feature_except(self, name_rlike: str = None, field: str = None, tower: str = None, deal: str = None, train_only: bool = True):
        return [f for f in list(self.train_features if train_only else self.features) if except_feature(f, name_rlike, field, tower, deal)]

    def get_features_by_name(self, names: List[str] = None, prefix: str = "", suffix: str = ""):
        if names:
            return [feature for feature in self.train_features if feature.name in names]
        elif prefix:
            return [feature for feature in self.train_features if feature.name.startswith(prefix)]
        elif suffix:
            return [feature for feature in self.train_features if feature.name.endswith(suffix)]
        else:
            raise ValueError("Names, prefix or suffix must given only one.")

    def __set_attr_by_deal_fun(self):
        for deal in FeatureDeal.__members__.values():
            if deal != FeatureDeal.Null:
                self.__setattr__(f"{deal.value}_features", self.get_deal_features(deal.value))

    def __set_feature_status(self, name: str = "", field: str = "", status: bool = True):
        assert name or field, "Name or field must given at least one of them"
        self.__check_feature_exist(name)
        for feature in self.features:
            if name and feature.name == name:
                feature.working = status
            elif field and feature.field_name == field:
                feature.working = status

    def set_feature_valid(self, name: str = "", field: str = ""):
        self.__set_feature_status(name, field, status=True)

    def set_feature_invalid(self, name: str = "", field: str = ""):
        self.__set_feature_status(name, field, status=False)

    def contains(self, name: str) -> bool:
        return len([f for f in self.features if f.name == name]) != 0

    def contains_field(self, field: str) -> bool:
        return len([f for f in self.features if f.field_name == field]) != 0

    def get_image_features(self):
        return self.get_deal_features("image")

    def get_embedding_features(self):
        return self.get_deal_features("embedding")


def filter_feature(feature: Feature, flag: str = None, field: str = None, tower: str = None, deal: str = None):
    """竖线代表 或逻辑
    """
    ret = True
    if flag and [f for f in flag.split("|") if f not in feature.name]:
        ret = False
    if tower and [t for t in tower.split("|") if feature.tower != FeatureTower(t)] :
        ret = False
    if deal and [d for d in deal.split("|") if feature.tower != FeatureDeal(d)]:
        ret = False
    if field and [f for f in field.split("|") if feature.field_name != f]:
        ret = False
    return ret


def except_feature(feature: Feature, flag: str = None, field: str = None, tower: str = None, deal: str = None):
    """竖线代表 或逻辑
    """
    ret = True
    if flag and [f for f in flag.split("|") if f in feature.name]:
        ret = False
    if tower and [t for t in tower.split("|") if feature.tower == FeatureTower(t)]:
        ret = False
    if deal and [d for d in deal.split("|") if feature.tower == FeatureDeal(d)]:
        ret = False
    if field and [f for f in field.split("|") if feature.field_name == f]:
        ret = False
    return ret
