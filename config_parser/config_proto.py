from enum import Enum


class FeatureTower(Enum):
    Null = "null"
    User = "user"
    Ad = "ad"
    Context = "context"
    Label = "label"


class FeatureDeal(Enum):
    Null = "null"
    Numeric = "numeric"
    Discrete = "discrete"
    Hashing = "hashing"
    Lookup = "lookup"
    StringLookup = "str_lookup"
    IntegerLookup = "int_lookup"
    Image = "image"  # 图像序列化，目前只支持tensorflow自带的图像序列化和反序列化
    Embedding = "embedding"  # embedding序列化，目前也只支持numpy和tf自带的序列化和反序列化
    TokenId = "token_id"  # token_id是事先将其encode成id了
    BertEncode = "bert_encode"  # bert_encode输入是str，现场将其encode成id


class FeaturePooling(Enum):
    Null = "null"
    Avg = "avg"
    Min = "min"
    Max = "max"
    Sum = "sum"
    Cls = "cls"
