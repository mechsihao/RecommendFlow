"""
创建人：MECH
创建时间：2021/12/19
功能描述：解析yaml配置文件，解析配置文件中的配置并替换配置文件中的变量替换符为'$'，因此需要避免正常的字符串中有这个符号
"""

from typing import Any, Dict, List
import pandas as pd
import yaml
from config_parser.config_utils import is_punctuation
from config_parser.features import Features
from utils.str_parser import str2dict, str2list


class Configuration(object):
    """
    整个配置层级一共有以下几种主要的层级结构
        - Features
        - Networks
        - Datasets
        - Experiments
        - ...
    """
    def __init__(self, config_path):
        self.conf = yaml.load(open(config_path).read(), Loader=yaml.FullLoader)

        self._init_global_conf()
        self._rematch_global_conf()

        self.features = Features(self.conf, self.get_conf_value("vocabs"), self.get_conf_value("seeds"))

        self.networks = self.conf["Networks"] if 'Networks' in self.conf else {}
        self.exp_conf = self.conf["Experiments"] if 'Experiments' in self.conf else None

        if not self.exp_conf:
            self.experiment_field = []
            self.experiments = pd.DataFrame()
        else:
            exp_fields = self.exp_conf["experiment_fields"]
            self.experiment_field = str2list(exp_fields) if isinstance(exp_fields, str) else exp_fields
            assert self.experiment_field[0] == "exp_id", "The first field must be exp_id"
            experiments = pd.DataFrame([self._parse_exp(exp) for exp in self.exp_conf["experiments"]], columns=self.experiment_field)
            self.experiments = experiments.set_index("exp_id")

    @property
    def train_features(self):
        return self.features.train_features

    @property
    def train_feature_names(self):
        return self.features.train_feature_names

    def _parse_exp(self, exp_conf: List[str]):
        try:
            exp_id = int(exp_conf[0])
        except Exception as e:
            raise Exception(f"Experiment first col must be integer type exp_id, got {type(exp_conf[0]).__name__}, detail: {str(e)}")

        content = [exp_id]
        for e in exp_conf[1:]:
            if not isinstance(e, str):
                content.append(e)
            else:
                if e.startswith("{") and e.endswith("}"):
                    content.append(str2dict(e[1:-1]))
                elif e.startswith("[") and e.endswith("]"):
                    content.append(str2list(e[1:-1], sep=";"))
                elif e.startswith("(") and e.endswith(")"):
                    content.append(str2list(e[1:-1], sep=";"))
                else:
                    content.append(self._set_str(e))
        return content

    def active_experiment(self, exp_id):
        """
        激活实验配置，如果配置了features这一列，则会修改原features中配置特征的状态
        :param exp_id: 实验id
        :return: 该实验对应的配置
        """
        if "features" in self.experiments.columns:
            features = self.experiments.loc[exp_id]["features"]
            assert isinstance(features, list), "Experiments field features must be a feature name list."
            for exp in features:
                # 优先处理name层级，其次处理field层级
                if exp[0] == "+":
                    if self.features.contains(exp[1:]):
                        self.features.set_feature_valid(name=exp[1:])
                    else:
                        self.features.set_feature_valid(field=exp[1:])
                elif exp[0] == "-":
                    if self.features.contains(exp[1:]):
                        # 优先处理name层级，其次处理field层级
                        self.features.set_feature_invalid(name=exp[1:])
                    else:
                        self.features.set_feature_invalid(field=exp[1:])
                else:
                    raise ValueError("Feature first latter must be '+/-' represent feature valid/invalid.")
        return self.experiments.loc[exp_id].to_dict()

    def get_conf_value(self, key: str, dtype: type = None):

        def __get_dict_value(dic: Dict[str, Any], k):
            res = None
            if k in dic:
                res = dic.get(k)
            else:
                for v in dic.values():
                    if isinstance(v, dict):
                        res = __get_dict_value(v, k)
                        if res is not None:
                            break
            return res

        ret = __get_dict_value(self.conf, key)
        if ret is None:
            raise KeyError(f"Could not find key='{key}' in configuration.")
        else:
            return dtype(ret) if dtype else ret

    def _set_value(self, v: Any):
        """将以$开头的整个字符串替换成global字典中的变量
        """
        if not isinstance(v, str):
            return v
        else:
            single_symbol = len([i for i in v if is_punctuation(i, except_char="_$")]) == 0
            if single_symbol and v.startswith("$"):
                return self.get_conf_value(v[1:])
            elif "$" in v:
                return self._set_str(v)
            else:
                return v

    def _set_str(self, v: Any):
        """将字符串中的$xxx子串替换成global字典中的变量，注意，$xxx以除了下划线'_'以外的标点符号分割子串，只能替换$后的一个子串
        若发现替换出来的内容非字符串，则直接报错
        """
        SPECIAL_SEP = "_##_"
        if not isinstance(v, str):
            return v
        else:
            string = ""
            for i in str(v):
                if i == "$":
                    string += SPECIAL_SEP + "$"
                elif is_punctuation(i, "_$"):
                    string += SPECIAL_SEP + i
                else:
                    string += i
            ret = []
            for i in string.split(SPECIAL_SEP):
                map_value = self.get_conf_value(i[1:]) if i.startswith("$") else i
                if isinstance(map_value, str) or isinstance(map_value, int) or isinstance(map_value, float) or isinstance(map_value, bool):
                    ret.append(str(map_value))
                else:
                    raise Exception(f"'$' symbol in sub string only support [str, int, float, bool], got {type(map_value).__name__}. "
                                    f"map_value: {map_value}.")
            return "".join(ret)

    def _init_global_conf(self):
        self.conf["Features"]["features"] = [[i for i in line.split(",")] for line in self.conf["Features"]["features"].split()]
        self.conf["Experiments"]["experiments"] = [[i for i in line.split(",")] for line in self.conf["Experiments"]["experiments"].split()]

    def _rematch_global_conf(self):
        """
        除掉features和experiment外，将其他的k-v都装进这个全局dict中，并且将其:
            - $开头的全部用已有的替换，若发现没有替换的告警，
            - 若发现key重名的抛出告警
        :return:
        """

        def __recurrent_set_list(input_list: List[Any]):
            res = []
            for i in input_list:
                if isinstance(i, list):
                    res.append(__recurrent_set_list(i))
                elif isinstance(i, dict):
                    res.append(_init_global_conf(i))
                else:
                    res.append(self._set_value(i))
            return res

        def _init_global_conf(conf):
            for k, v in conf.items():
                if isinstance(v, dict):
                    _init_global_conf(v)
                elif isinstance(v, list):
                    res = []
                    for i in v:
                        sub_i = self._set_value(i)
                        if isinstance(sub_i, int) or isinstance(sub_i, str) or isinstance(sub_i, float):
                            res.append(sub_i)
                        elif isinstance(sub_i, list):
                            res.append(__recurrent_set_list(sub_i))
                        else:
                            raise ValueError(f"'$' symbol in list must be [str, int, float], got {type(sub_i).__name__}, sub_i: {sub_i}")
                    conf[k] = res
                else:
                    conf[k] = self._set_value(v)

        _init_global_conf(self.conf)

    def _set_conf(self):
        """
        除掉features和experiment外，将其他的k-v都装进这个全局dict中，并且将其:
            - $开头的全部用已有的替换，若发现没有替换的告警，
            - 若发现key重名的抛出告警
        :return:
        """
        for k, v in self.conf.items():
            if k not in ["Features", "Train", "Evaluate", "Infer"] or k == "Experiments" and v not in ["experiment_fields", "experiments"]:
                if k in self.conf:
                    raise Exception(f"There is conflicts name for key = {k}, value = {v}, it wasn't allowed.")
                else:
                    if isinstance(v, str):
                        self.conf[k] = self._set_value(v)
                    elif isinstance(v, list):
                        self.conf[k] = [self._set_value(i) for i in v]
                    elif isinstance(v, dict):
                        for k_sub, v_sub in v.items():
                            if k_sub in self.conf:
                                raise Exception(f"There is conflicts name for key = {k_sub}, value = {v_sub}, it wasn't allowed.")
                            else:
                                if isinstance(v_sub, list):
                                    self.conf[k_sub] = [self._set_value(i) for i in v_sub]
                                else:
                                    self.conf[k_sub] = self._set_value(v_sub)

    def print_features(self, scale: str = "train", blank_size: int = 2):

        def cal_blank(max_x, x):
            return max_x - len(x) + blank_size

        def record_max(tmp_score, max_score):
            if tmp_score > max_score:
                max_score = tmp_score
            return max_score

        res, max_name, max_field, max_tower, max_deal, max_ftype, max_is_valid = [], 0, 0, 0, 0, 0, 0
        for i, f in enumerate(self.features.features) if scale == "all" else enumerate(self.train_features):
            name, field, tower, deal, ftype, is_valid = f"name={f.name}", f"field={f.field_name}", f"tower={f.tower.value}", \
                                                        f"deal={f.deal.value}", f"type={f.type.name}", f"working={f.working}"

            max_name = record_max(len(name), max_name)
            max_field = record_max(len(field), max_field)
            max_tower = record_max(len(tower), max_tower)
            max_deal = record_max(len(deal), max_deal)
            max_ftype = record_max(len(ftype), max_ftype)
            max_is_valid = record_max(len(is_valid), max_is_valid)

            res.append([name, field, tower, deal, ftype, is_valid])

        for i, line in enumerate(res):
            name, field, tower, deal, ftype, is_valid = line

            info = ""
            info += name + " " * cal_blank(max_name, name)
            info += field + " " * cal_blank(max_field, field)
            info += tower + " " * cal_blank(max_tower, tower)
            info += deal + " " * cal_blank(max_deal, deal)
            info += ftype + " " * cal_blank(max_ftype, ftype)
            info += is_valid

            print(f"Feature {i}:\t[{info}]")
