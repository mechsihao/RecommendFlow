"""
创建人：MECH
创建时间：2022/06/01
功能描述：faiss索引构建检索系统的全流程
更新记录：
    - 20221220 新增支持纯向量构建searcher，不需要encoder
    - 20230102 去除冗余功能单独从原始项目中剥离
"""
import os
import time

from typing import List, Union, Dict, Any, Tuple
import numpy as np
import pandas as pd
import faiss

from pandas import DataFrame
from numpy import array, ndarray
import tensorflow as tf
import pickle


class FaissSearcher:
    """
    faiss索引构建检索系统的全流程
    """
    def __init__(
            self,
            encoder: Any = None,
            items: Union[DataFrame, ndarray] = None,
            item_list: List[Any] = None,
            index_param: str = None,
            measurement: Union[str, int] = None,
            norm_vec: bool = False,
            use_gpu: bool = False,
            **kwargs
    ):
        if encoder is None:
            self.encoder = None
            if not isinstance(items, ndarray):
                raise ReferenceError("如果不传入encoder，则item只能输入numpy.array类型")
            if item_list is not None:
                assert len(item_list) == len(items), f"len(item_list)={len(item_list)} != len(items)={len(items)}"
        else:
            # 兼容通用encoder
            # 兼容通用encoder
            if hasattr(encoder, "encode"):
                self.encoder = encoder
            else:
                raise AttributeError("传入的encoder必须包含predict方法...")

        assert items is not None and index_param is not None and measurement is not None, "Args 'items' 'index_param' 'measurement' " \
                                                                                          "must be given."
        self.index_param = index_param
        self.items = items
        self.item_list = np.array(item_list)
        self.norm_vec = True if measurement == 'cos' else norm_vec
        self.metric = self.set_measure_metric(measurement)
        self.measurement = measurement
        self.vec_dim = self.get_vecs(items[items.columns[0]][:1].to_list(), verbose=0).shape[1] if encoder else items.shape[1]
        self.vecs = None
        self.index = None
        self.gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        self.use_gpu = use_gpu and self.__check_faiss_gpu()
        self.kwargs = kwargs

    def get_vecs(self, items: Union[List[str], ndarray], verbose: int = 1) -> array:
        if self.encoder:
            vecs = self.encoder.encode(items, verbose=verbose)
        else:
            assert isinstance(items, ndarray), "encoder=None, items必须输入向量矩阵，类型为ndarray"
            assert len(items.shape) == 2, f"encoder=None, 输入只能是二维矩阵[(n, dim)]，当前维度为[{items.shape}]"
            vecs = items
        if self.norm_vec:
            return self.__tofloat32__(self.__normvec__(vecs))
        else:
            return self.__tofloat32__(vecs)

    @staticmethod
    def set_measure_metric(measurement):
        metric_dict = {
            'ip': faiss.METRIC_INNER_PRODUCT,
            'cos': faiss.METRIC_INNER_PRODUCT,
            'l1': faiss.METRIC_L1,
            'l2': faiss.METRIC_L2,
            'l_inf': faiss.METRIC_Linf,
            'l_p': faiss.METRIC_Lp,
            'brayCurtis': faiss.METRIC_BrayCurtis,
            'canberra': faiss.METRIC_Canberra,
            'jensen_shannon': faiss.METRIC_JensenShannon
        }
        if measurement in metric_dict:
            return metric_dict[measurement]
        elif isinstance(measurement, int):
            return measurement
        else:
            raise Exception(f"Do not support measurement: '{measurement}', support measurement is [{', '.join(list(metric_dict.keys()))}]")

    @staticmethod
    def __tofloat32__(vecs):
        return vecs.astype(np.float32)

    @staticmethod
    def __normvec__(vecs):
        return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5

    def __build_faiss_index(self):
        """
        Faiss必须是1.7.3以上，cpu和gpu均可
        如果是旧版本faiss构建hnsw需要做以下处理（官方的小bug）：
        >>>  if 'hnsw' in self.index_param.lower() and ',' not in self.index_param:
        >>>      self.index = faiss.IndexHNSWFlat(self.vec_dim, int(self.index_param.split('HNSW')[-1]), self.metric)
        >>>  else:
        >>>      self.index = faiss.index_factory(self.vec_dim, self.index_param, self.metric)
        :return:
        """
        self.index = faiss.index_factory(self.vec_dim, self.index_param, self.metric)
        self.index.verbose = True
        self.index.do_polysemous_training = False
        if self.use_gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index)
        return self

    def __check_faiss_gpu(self):
        assert self.gpus, "No gpu devices found. please check or set `use_gpu=False`."
        package_info = [i.split() for i in os.popen("pip3 list | grep faiss-gpu").read().split('\n') if i]
        if len(package_info) == 0:
            raise Exception("faiss-gpu未安装，请先安装faiss-gpu")
        else:
            return True

    def load_index(self, index_path):
        print(f"Load index...")
        self.index = faiss.read_index(index_path)
        assert self.index.ntotal == len(self.items), f"Index sample nums {self.index.ntotal} != Items length {len(self.items)}"
        assert self.index.d == self.vec_dim, f"Index dim {self.index.d} != Vecs dim {self.vec_dim}"
        assert self.index.is_trained, "Index dose not trained"
        if self.use_gpu:
            self.index = faiss.index_cpu_to_all_gpus(self.index)

    def train(self):
        print(f"Encode items start...")
        self.vecs = self.get_vecs(self.items[self.items.columns[0]].to_list() if self.encoder else self.items)
        start_time = time.time()
        vecs = self.__tofloat32__(self.vecs)
        print(f"Train index start...")
        self.__build_faiss_index()
        self.index.train(vecs)
        self.index.add(vecs)
        print(f"Train index cost time: {time.time() - start_time}")

    def search_items(self, target: List[str], indexes: ndarray, directories: ndarray, keep_rank_no: bool = False) -> \
            Union[Tuple[ndarray, ndarray], Tuple[ndarray, ndarray, ndarray], DataFrame]:
        """
        返回df列名：["source_item", "sim_item", "sim_val" + 原items除第一列外的剩下的列]
        """
        start_time = time.time()
        if not self.encoder and keep_rank_no:
            res = (self.item_list[indexes], directories[indexes], indexes)
        elif not self.encoder and not keep_rank_no:
            res = (self.item_list[indexes], directories)
        else:
            target = pd.DataFrame(target, columns=['source_item'])
            target['sim_ind'] = list(indexes)
            target['sim_val'] = list(directories)
            target['pair'] = target.apply(lambda x: [[c[0], c[1], i] for i, c in enumerate(zip(x['sim_ind'], x['sim_val']))], axis=1)
            target = target.drop(columns=['sim_ind', 'sim_val'])
            target = target.explode('pair').reset_index(drop=True)
            target[['sim_ind', 'sim_val', 'rank_no']] = pd.DataFrame(target['pair'].to_list(), columns=['sim_ind', 'sim_val', 'rank_no'])
            target['sim_val'] = target['sim_val'].values.astype(np.float32)
            sim_item = self.items.iloc[target['sim_ind']].reset_index(drop=True)
            sim_item.columns = ['sim_item'] + list(sim_item.columns[1:])
            target = target.drop(columns=['pair', 'sim_ind']) if keep_rank_no else target.drop(columns=['pair', 'sim_ind', 'rank_no'])
            res = pd.concat([target, sim_item], axis=1)
        print(f"Find items cost time: {time.time() - start_time}")
        return res

    def search(self, target: Union[List[str], ndarray], topK: Union[int, List[int]],
               keep_rank_no=False) -> Union[DataFrame, Dict[int, DataFrame], Tuple[ndarray, ndarray], Tuple[ndarray, ndarray, ndarray]]:
        if not self.index:
            raise Exception("Faiss dose not train, please use train method before search or load a trained index...")
        else:
            target_vec = self.get_vecs(target)
            if isinstance(topK, int):
                start_time = time.time()
                directories, indexes = self.index.search(target_vec, topK)
                print(f"Search index cost time: {time.time() - start_time}")
                return self.search_items(target, indexes, directories, keep_rank_no=keep_rank_no)
            elif isinstance(topK, List):
                start_time = time.time()
                res = {}
                directories, indexes = self.index.search(target_vec, max(topK))
                print(f"Search index topK={max(topK)} cost time: {time.time() - start_time}")
                max_res = self.search_items(target, indexes, directories, keep_rank_no=True)
                for k in topK:
                    if self.encoder:
                        tmp_res = max_res.query(f"rank_no < {k}").reset_index(drop=True)
                        res[k] = tmp_res if keep_rank_no else tmp_res.drop(columns=['rank_no'])
                    else:
                        new_item, new_sim, new_ind = max_res[0][:, :k], max_res[1][:, :k], max_res[2][:, :k]
                        res[k] = (new_item, new_sim, new_ind) if keep_rank_no else (new_item, new_sim)
                return res
            else:
                raise TypeError(f"TopK dose not support type: {type(topK)}")

    def save_index(self, index_save_path):
        faiss.write_index(self.index, index_save_path)

    def cal_sim(self, item1: str, items2: List[str]):
        vec1 = self.encoder.encode([item1], verbose=0)
        vecs2 = self.encoder.encode(items2, verbose=0)
        sim_score_list = list(vec1.dot(vecs2.T))
        sim_df = pd.DataFrame([items2], columns=['item'])
        sim_df['score'] = sim_score_list
        return sim_df.sort_values(by="score", ascending=False)

    def save_searcher(self, path):
        file = open(path, "wb")
        pickle.dump(self, file)
        file.close()

    @staticmethod
    def load_searcher(path):
        file = open(path, "rb")
        return pickle.load(file)
