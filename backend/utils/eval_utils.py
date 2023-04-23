import csv
import re
import time
from typing import List, Dict, Tuple, Union

import scipy.stats
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn import metrics
from sklearn.metrics import precision_recall_curve, roc_auc_score, mean_squared_error

from backend.third_party_components.faiss_searcher import FaissSearcher


def split_by_type(text):
    regex = r"[\u4e00-\u9fa5]+|[0-9]+|[a-zA-Z]+|[ ]+|[.#]+"
    return re.findall(regex, text, re.UNICODE)


def clean_text(text):
    return ''.join(split_by_type(' '.join(text.lower().split())))


def load_id_name_map(file_name):
    df = pd.read_csv(file_name, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8', on_bad_lines='skip',
                     converters={i: str for i in range(100)})
    column_id, column_name = df.columns[0:2]
    df[column_name] = df[column_name].apply(clean_text)
    return df.set_index(column_id)[column_name].to_dict()


def calculate_auc(label, predict):
    fpr, tpr, threshold = metrics.roc_curve(label, predict)
    return metrics.auc(fpr, tpr)


def calculate_aupr(label, predict):
    precision, recall, threshold = metrics.precision_recall_curve(label, predict)
    return metrics.auc(recall, precision)


def calculate_pr_at_precision(label, predict, min_precision):
    precision, recall, threshold = metrics.precision_recall_curve(label, predict)
    satisfied_metric = []
    for th, p, r in zip(threshold, precision, recall):
        if p >= min_precision:
            satisfied_metric.append([th, p, r])
    if len(satisfied_metric) == 0:
        # the last precision and the recall value should be ignored.
        # The last precision and recall values are always 1. and 0. respectively
        # and do not have a corresponding threshold.
        max_precision_index = precision.tolist().index(max(precision.tolist()[:-1]))
        return threshold[max_precision_index], precision[max_precision_index], 'no_recall'
    else:
        return sorted(satisfied_metric, key=(lambda x: x[2]), reverse=True)[0]


def calculate_metrics(eval_df, min_precision):
    label = eval_df['label'].tolist()
    score = eval_df['score'].tolist()
    auc_score = calculate_auc(label, score)
    aupr = calculate_aupr(label, score)
    threshold, precision, recall = calculate_pr_at_precision(label, score, min_precision)
    return auc_score, aupr, threshold, precision, recall


def evaluate_recall_at_precision(eval_df, sample_items, min_precision=0.6):
    total_auc, total_aupr, total_threshold, total_precision, total_recall = calculate_metrics(eval_df, min_precision)
    items_auc, items_aupr, items_threshold, items_precision, items_recall = calculate_metrics(
        eval_df[eval_df['item'].isin(sample_items)], min_precision
    )
    return (
        total_auc, total_aupr, total_threshold, total_precision, total_recall,
        items_auc, items_aupr, items_threshold, items_precision, items_recall
    )


def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation


def get_click_index(rec_id: ndarray, label_id: ndarray) -> List[float]:
    """获取label在推荐矩阵中的命中位置向量，如果某条记录没有命中，则该记录处会被标记上一个无穷大的正数，方便后续计算 xxx@K
    :param rec_id: 推荐矩阵，一般维度是 [n, max(topk_list)]
    :param label_id: 真实点击array，一维数组，维度是 (n, )
    """
    assert rec_id.shape[0] == label_id.shape[0], "Sample nums does not match!"
    click_metrics = np.array(rec_id == label_id[:, None]).astype(int).argmax(axis=1)
    # pad mat的目的是消除掉那些没有匹配到label，但是由于argmax会返回0的，令其置为一个最大的数max(topk_list) + 1
    # 先找到第0列不等于label_id的所有index
    pad_index_mat = np.array(~(np.ones_like(rec_id[:, 0]) == label_id)).astype(int)
    # 再给click_metrics的这些不为0的index处的值附上一个比较大的数
    pad_mat = ((click_metrics == 0).astype(int) * pad_index_mat) * int(1e14)
    # 最后将该arr和点击arr加起来
    click_metrics += pad_mat
    return click_metrics


def batch_get_click_ids(searcher, targets, labels, batch_size, k):
    start_search = time.time()
    click_id_list = []
    n = len(targets) // batch_size + 1
    for i in range(n):
        tmp_vecs = targets[i * batch_size: (i + 1) * batch_size]
        tmp_labels = labels[i * batch_size: (i + 1) * batch_size]
        if len(tmp_vecs) == 0:
            pass
        else:
            rec_ids = searcher.search(tmp_vecs, k)
            tmp_click_index = get_click_index(rec_ids[0], tmp_labels)
            click_id_list.append(tmp_click_index)
    click_ids = np.hstack(click_id_list)
    print(f"Batch search recall cost: {time.time() - start_search}")
    return click_ids


def batch_compute_recall_score(searcher: FaissSearcher,
                               targets: ndarray,
                               labels: ndarray,
                               topk_list: List[int],
                               weights: ndarray,
                               batch_size: int) -> Tuple[List[float], List[float], List[float]]:
    """为防止一次向量加载过多，允许分批计算hit、mrr、ndcg
    :param searcher: 检索器
    :param targets: 目标向量/目标item，总之就是直接能被searcher进行检索的
    :param labels: 真实点击array，一维数组，维度是 (n, )
    :param topk_list: topK list
    :param weights 每条样本的加权数组
    :param batch_size: batch大小
    """
    click_ids = batch_get_click_ids(searcher, targets, labels, batch_size, max(topk_list))

    start_eval = time.time()
    hit, mrr, ndcg = [], [], []
    for k in topk_list:
        info = (click_ids < k).astype(int)
        dcgs = 1 / np.log2(click_ids + 2) * info
        i_dcgs = 1 / np.log2(info + 2) * info + 1e-12  # 本来的i_dcgs计算是将真实打分倒序排列计算dcg、但是在我们这个场景里真实打分最多只有1个，并且分数就是1，因此idcg计算简化
        hit.append((info * weights).sum() / (weights.sum() + 1e-12))
        mrr.append(((1 / (click_ids + 1)) * weights).sum() / (weights.sum() + 1e-12))
        ndcg.append(((dcgs / i_dcgs) * weights).sum() / (weights.sum() + 1e-12))

    print(f"Batch evaluate all cost: {time.time() - start_eval}")
    return hit, mrr, ndcg


def batch_compute_group_recall_score(searcher: FaissSearcher,
                                     targets: ndarray,
                                     labels: ndarray,
                                     group_ids: Union[ndarray, None],
                                     topk_list: List[int],
                                     weights: ndarray,
                                     batch_size: int) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    """评估召回
    :param searcher: 检索器
    :param targets: 目标向量/目标item，总之就是直接能被searcher进行检索的物料，和searcher强相关
    :param labels: 真实点击array，一维向量，维度是 (n, )
    :param group_ids: 分组id向量，维度是 (n, )，支持分组评估recall指标
    :param topk_list: topK list
    :param weights 每条样本的加权数组，一般都是np.ones，可以用在两个场景下：
        - 去重评估，用来节省空间和时间，weights可以是每条记录的重复次数
        - 重要性不同的评估，weights可以是每条样本不同的重要性权重
    :param batch_size: 每次处理batch大小：
        - batch越大速度越快，但是内存要求越多
        - batch越小速度越慢，但是内存要求越少
        - k越大，需要适当调小batch，防止爆内存
    """
    click_ids = batch_get_click_ids(searcher, targets, labels, batch_size, max(topk_list))

    def insert_dic(input_dict, key, val):
        if key in input_dict:
            input_dict[key].append(val)
        else:
            input_dict[key] = [val]

    start_eval = time.time()
    group_hit, group_mrr, group_ndcg = {}, {}, {}
    for k in topk_list:
        info = (click_ids < k).astype(int)
        dcgs = 1 / np.log2(click_ids + 2) * info
        i_dcgs = 1 / np.log2(info + 2) * info + 1e-12
        hit = (info * weights).sum() / (weights.sum() + 1e-12)
        mrr = ((1 / (click_ids + 1)) * weights).sum() / (weights.sum() + 1e-12)
        ndcg = ((dcgs / i_dcgs) * weights).sum() / (weights.sum() + 1e-12)
        insert_dic(group_hit, "all", hit)
        insert_dic(group_mrr, "all", mrr)
        insert_dic(group_ndcg, "all", ndcg)

        if group_ids is not None:
            for group_id in list(set(group_ids)):
                idx = np.array(group_ids == group_id).astype(int)
                tmp_hit = (info * weights * idx).sum() / ((weights * idx).sum() + 1e-12)
                tmp_mrr = ((1 / (click_ids + 1)) * weights * idx).sum() / ((weights * idx).sum() + 1e-12)
                tmp_ndcg = ((dcgs / i_dcgs) * weights * idx).sum() / ((weights * idx).sum() + 1e-12)
                insert_dic(group_hit, group_id, tmp_hit)
                insert_dic(group_mrr, group_id, tmp_mrr)
                insert_dic(group_ndcg, group_id, tmp_ndcg)

    print(f"Batch evaluate all cost: {time.time() - start_eval}")
    return group_hit, group_mrr, group_ndcg


def get_recall_eval_info(result_dict: Dict[str, Dict[str, List[float]]], topk_list):
    assert len(result_dict), "At least contains one eval metric, got nothing"
    group_ids = list(list((result_dict.values()))[0].keys())
    # 读取计算指标
    all_info_strs = []
    for i, group in enumerate(group_ids):
        group_info_strs = []
        for metric_name, metric_dict in result_dict.items():
            group_info_strs.append(", ".join([f"{metric_name}@{k}:{v:.5f}" for k, v in zip(topk_list, metric_dict[group])]))
        group_info_str = "\n".join(group_info_strs)
        all_info_strs.append((f"{i + 1}.group_id = {group}:\n", f"""{group_info_str}\n"""))
    if len(all_info_strs) == 1:
        return all_info_strs[0][1]
    else:
        return "\n".join([key + info for key, info in all_info_strs])


def compute_hit_score(rec_id: ndarray, label_id: ndarray, topk_list: List[int], weights: ndarray) -> List[float]:
    """命中率
    :param rec_id: 推荐矩阵，一般维度是 [n, max(topk_list)]
    :param label_id: 真实点击array，一维数组，维度是 (n, )
    :param topk_list:
    :param weights 每条样本的加权数组
    """
    assert rec_id.shape[0] == label_id.shape[0], "Sample nums does not match!"
    res = []
    for k in topk_list:
        click_metrics = np.array(rec_id[:, :k] == label_id[:, None]).astype(int)
        res.append((click_metrics.sum(1) * weights).sum() / (weights.sum() + 1e-12))
    return res


def compute_mrr_score(rec_id: ndarray, label_id: ndarray, topk_list: List[int], weights: ndarray) -> List[float]:
    """mrr
    :param rec_id: 推荐矩阵，一般维度是 [n, max(topk_list)]
    :param label_id: 真实点击array，一维数组，维度是 (n, )
    :param topk_list:
    :param weights 每条样本的加权数组
    """
    assert rec_id.shape[0] == label_id.shape[0], "Sample nums does not match!"
    res = []
    for k in topk_list:
        click_metrics = np.array(rec_id[:, :k] == label_id[:, None]).astype(int) / np.arange(1, k+1)
        res.append((click_metrics.sum(1) * weights).sum() / (weights.sum() + 1e-12))
    return res


def compute_ndcg_score(rec_id: ndarray, label_id: ndarray, topk_list: List[int], weights: ndarray) -> List[float]:
    """ndcg
    :param rec_id: 推荐矩阵，一般维度是 [n, max(topk_list)]
    :param label_id: 真实点击array，一维数组，维度是 (n, )
    :param topk_list:
    :param weights 每条样本的加权数组
    """
    assert rec_id.shape[0] == label_id.shape[0], "Sample nums does not match!"
    res = []
    for k in topk_list:
        i_dcgs = np.log2(np.arange(k) + 2)
        dcg = (np.array(rec_id[:, :k] == label_id[:, None]).astype(int) / i_dcgs).sum(1)
        idcg = (-np.sort(-np.array(rec_id[:, :k] == label_id[:, None]).astype(int)) / i_dcgs).sum(1)
        res.append(((dcg / (idcg + 1e-12)) * weights).sum() / (weights.sum() + 1e-12))
    return res


def recall_at_precision(y_true, y_score, min_precision=0.65):
    """固定 精度 求最大召回
    """
    p, r, th = precision_recall_curve(y_true, y_score)

    satisfied_metric = []
    # precision_recall_curve返回的threshold比pr少一维，这里填上对齐
    th = np.insert(th, 0, min(y_score))
    for th, p, r in zip(th, p, r):
        if p >= min_precision:
            satisfied_metric.append([th, p, r])

    if satisfied_metric:  # 线上要求: p>=0.8 and max(r)
        th_best, p_best, r_best = sorted(satisfied_metric, key=(lambda x: x[2]), reverse=True)[0]
    else:  # 如果没有满足线上要求的，最取precision最大值
        p_best = max(p)
        p_best_idx = p.tolist().index(p_best)
        r_best = r[p_best_idx]
        if p_best_idx >= len(th):
            th_best = th[-1]
        else:
            th_best = 0

    return r_best, p_best, th_best


def auc(y_true, y_pred):
    return roc_auc_score(y_true=y_true, y_score=y_pred)


def mse(y_true, y_pred):
    return mean_squared_error(y_true=y_true, y_pred=y_pred)
