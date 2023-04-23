import json
import os

import tensorflow.keras.backend as K
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import DataGenerator

import functools
from concurrent import futures

executor = futures.ThreadPoolExecutor(1)


def timeout(seconds):
    """超时装饰器，用于限制bert service的输出时间，如果bert service输入太慢，则认为其连接失败，退出执行本地encode, seconds单位是秒
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            future = executor.submit(func, *args, **kwargs)
            return future.result(timeout=seconds)
        return wrapper
    return decorator


def merge(inputs):
    """向量合并：a、b、|a-b|拼接
    """
    a, b = inputs[::2], inputs[1::2]
    o = K.concatenate([a, b, K.abs(a - b)], axis=1)
    # 这里需要repeat的原因是，该pair generator会将query和app叠在一起输入模型中，label被重复了一遍，为原来的两倍，
    # 如果直接用concat会产生dim不匹配的问题。因此要将其重复一遍，但是这样会让loss也多计算了一遍，不过影响很像小
    return K.repeat_elements(o, 2, 0)


class EncodeDataGenerator(DataGenerator):
    """特征式数据生成器，输入格式为：[text]
    """

    def __init__(self, data, dict_path, max_len=32, batch_size=32, buffer_size=None):
        super(EncodeDataGenerator, self).__init__(data, batch_size=batch_size, buffer_size=buffer_size)
        self.max_len = max_len
        self.dict_path = dict_path
        self.tokenizer = Tokenizer(self.dict_path, do_lower_case=True)

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, text in self.sample(random):
            token_id, segment_id = self.tokenizer.encode(str(text), maxlen=self.max_len)
            batch_token_ids.append(token_id)
            batch_segment_ids.append(segment_id)
            batch_labels.append([0])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    def forpred(self):
        while True:
            for d in self.__iter__():
                yield d


def load_config(config_path="./bert_service_conf.json"):
    """读取json格式配置
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    return config
