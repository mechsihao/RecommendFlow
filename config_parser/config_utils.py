import json
import os
import unicodedata
from typing import Dict, Any


def load_config(config_path):
    return parse_json_config(config_path, True)


def save_config(config_path, conf):
    """保存json格式配置
    """
    with open(config_path, 'w') as f:
        json.dump(conf, f)


def print_conf(conf_path):
    """打印json配置文件
    """
    def make_dict_list_to_str(a: Dict[str, Any]):
        for k, v in a.items():
            if isinstance(v, list):
                a[k] = [", ".join([str(i) for i in v])]
            if isinstance(v, dict):
                make_dict_list_to_str(a[k])
        return a

    conf = parse_json_config(conf_path, mode='print')
    res = json.dumps(make_dict_list_to_str(conf), indent=2, ensure_ascii=False)
    print(res)


def parse_del_features(conf, mode):
    """在所有list中的字段，如果前面加上#，则代表删除该字段
    """
    def del_dict_list_features(a: Dict[str, Any], name: str):
        for k, v in a.items():
            if isinstance(v, list):
                del_cols = "', '".join([str(i)[1:] for i in v if str(i).startswith("#")])
                a[k] = [i for i in v if not str(i).startswith("#")]
                if del_cols and mode == 'load':
                    print(f"[Config Warning] level='{name}', key='{k}', delete values: '{del_cols}'")
                if v and not a[k]:
                    raise ValueError(f"[Config Error] level='{name}', key='{k}', can't delete all features!")
            if isinstance(v, dict):
                del_dict_list_features(a[k], k)
    del_dict_list_features(conf, "Config")


def parse_json_config(config_path, non_exist_raise_error: bool = True, mode: str = "load"):
    """读取json格式配置
    """
    assert mode in ('print', 'load')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        if non_exist_raise_error:
            raise FileNotFoundError(f"[Error]未发现配置文件：{config_path}")
        else:
            config = {}
    parse_del_features(config, mode)  # 解析被删除的字段并将其从配置中删除
    return config


def is_punctuation(ch, except_char: str = ""):
    """标点符号类字符判断（全/半角均在此内）
    提醒：unicodedata.category这个函数在py2和py3下的
    表现可能不一样，比如u'§'字符，在py2下的结果为'So'，
    在py3下的结果是'Po'。
    """
    if ch in except_char:
        return False
    code = ord(ch)
    return 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126 or \
        unicodedata.category(ch).startswith('P')


def load_vocab(dict_path, encoding='utf-8'):
    """从bert的词典文件中读取词典
    """
    token_dict = {}
    with open(dict_path, encoding=encoding) as reader:
        for line in reader:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)
    return token_dict
