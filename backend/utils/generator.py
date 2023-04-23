from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.tokenizers import Tokenizer


class PairTextGenerator(DataGenerator):
    """
    数据生成器，数据格式 [text1 text2 label]
    """
    def __init__(self, data, dict_path, batch_size, max_len, buffer_size=None):
        super(PairTextGenerator, self).__init__(data, batch_size, buffer_size)
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)
        self.max_len = max_len

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            for text in [str(text1), str(text2)]:
                token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.max_len)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([float(label)])

            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    def forpred(self):
        while True:
            for d in self.__iter__():
                yield d


class PairInteractTextGenerator(DataGenerator):
    """
    数据生成器，数据格式 [text1 text2 label]，输出的label的数量为 text数量的一半
    """
    def __init__(self, data, dict_path, batch_size, max_len, buffer_size=None):
        super(PairInteractTextGenerator, self).__init__(data, batch_size, buffer_size)
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)
        self.max_len = max_len

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            for text in [str(text1), str(text2)]:
                token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.max_len)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)

            batch_labels.append([float(label)])

            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    def forpred(self):
        while True:
            for d in self.__iter__():
                yield d


class PairWeightTextGenerator(DataGenerator):
    """
    数据生成器，数据格式 [text1 text2 label weight]
    """
    def __init__(self, data, dict_path, batch_size, max_len, use_weight=False, buffer_size=None):
        super(PairWeightTextGenerator, self).__init__(data, batch_size, buffer_size)
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)
        self.max_len = max_len
        self.use_weight = use_weight

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels, batch_weights = [], [], [], []
        for is_end, line in self.sample(random):
            if self.use_weight:
                (text1, text2, label, weight) = line
            else:
                (text1, text2, label), weight = line, 1

            for text in [str(text1), str(text2)]:
                token_ids, segment_ids = self.tokenizer.encode(text, maxlen=self.max_len)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([float(label)])
                batch_weights.append([float(weight)])

            if len(batch_token_ids) == self.batch_size * 2 or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                batch_weights = sequence_padding(batch_weights)
                yield [batch_token_ids, batch_segment_ids, batch_weights], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels, batch_weights = [], [], [], []

    def forpred(self):
        while True:
            for d in self.__iter__():
                yield d


class InteractGenerator(DataGenerator):
    """
    数据生成器，句子格式为[text1 text2 label]，不同于PairTextGenerator的是，text1 text2用[SEP]拼接成一句话送入BERT中
    """
    def __init__(self, data, dict_path, batch_size, max_len, buffer_size=None):
        super(InteractGenerator, self).__init__(data, batch_size, buffer_size)
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)
        self.max_len = max_len

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(str(text1), str(text2), maxlen=self.max_len)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append([0] * len(segment_ids))
            batch_labels.append([float(label)])
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


class SimbertDataGenerator(DataGenerator):
    """
    数据生成器，输入数据格式为 [text1, text2]，都是相似句
    """
    def __init__(self, input_data, input_tokenizer, batch_size=32, max_len=32, buffer_size=None):
        super(SimbertDataGenerator, self).__init__(input_data, batch_size, buffer_size)
        self.some_samples = []
        self.max_len = max_len
        self.tokenizer = input_tokenizer

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, text_list in self.sample(random):
            if len(text_list) != 2:
                continue
            else:
                text1, text2 = text_list[0], text_list[1]
                self.some_samples.append(text1)
                if len(self.some_samples) > 1000:
                    self.some_samples.pop(0)

                token_ids, segment_ids = self.tokenizer.encode(
                    text1, text2, maxlen=self.max_len * 2
                )
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)

                token_ids, segment_ids = self.tokenizer.encode(
                    text2, text1, maxlen=self.max_len * 2
                )
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)

                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    yield [batch_token_ids, batch_segment_ids], None
                    batch_token_ids, batch_segment_ids = [], []

    def forpred(self):
        while True:
            for d in self.__iter__():
                yield d


class BaseGenerator(DataGenerator):
    """
    数据生成器，句子格式为[text1 label]
    """
    def __init__(self, data, dict_path, batch_size, max_len, buffer_size=None):
        super(BaseGenerator, self).__init__(data, batch_size, buffer_size)
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)
        self.max_len = max_len

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(str(text), maxlen=self.max_len)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([float(label)])
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


class Que2SearchDataGenerator(DataGenerator):
    """特征式数据生成器，输入格式为：[text]
    """
    def __init__(self, data_dict, dict_path, max_len=32, batch_size=32, buffer_size=None):
        super(Que2SearchDataGenerator, self).__init__(data_dict, batch_size=batch_size, buffer_size=buffer_size)
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
