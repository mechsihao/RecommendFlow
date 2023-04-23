import os
import argparse

import tensorflow as tf
from functools import partial

from config_parser.configuration import Configuration
from business.utils.qa_dataloader import get_dataset_dict_with_dayno
from business.utils.evaluator import Evaluator
from backend.utils.model_utils import build_network
from backend.core.dataloader import load_tfrecord_dayno_patten_split
from backend.utils.gpu_utils import auto_device_strategy, set_mem_growth_gpus

from utils.print_util import print_args_table
from utils.env_util import activate_hadoop_env
from utils.str_parser import str2dayno, str2loss, str2list, str2bool

from tensorflow.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

activate_hadoop_env()
set_mem_growth_gpus()


def arg_parser():
    parser = argparse.ArgumentParser(description='BertDssm Train')
    parser.add_argument('--task', type=str, default='browse_search')
    parser.add_argument('--conf_path', type=str,
                        default='/home/notebook/data/group/sihao_work/pip_rec/conf/base_conf.yaml', help='预训练模型加载路径')
    parser.add_argument('--train_tfr_root', type=str,
                        default='hdfs://ad-hdfs/hive-dw/ad/tag/recall_search/que2search/browse_search/daily_train/YYYYMMDD/'
                                'train_tfr_neg_10_exp5_ctr_002',
                        help='模型训练数据路径')
    parser.add_argument('--eval_tfr_root', type=str,
                        default='hdfs://ad-hdfs/hive-dw/ad/tag/recall_search/que2search/browse_search/daily_train/YYYYMMDD/train_tfr_neg0',
                        help='模型训练数据路径')
    parser.add_argument('--smt_tfr_root', type=str,
                        default='hdfs://ad-hdfs/hive-dw/ad/tag/recall_search/que2search/browse_search/daily_semantic/YYYYMMDD/smt_tfr',
                        help='模型评估数据路径')
    parser.add_argument('--app_cand_tfr_root', type=str,
                        default='hdfs://ad-hdfs/hive-dw/ad/tag/recall_search/que2search/browse_search/daily_app_cand/YYYYMMDD/app_cand_tfr',
                        help='app数据路径')
    parser.add_argument('--sample_ratio', type=float, default='1', help='数据采样比例')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=32, help='batch大小')
    parser.add_argument('--prefetch_prob', type=float, default=5, help='prefetch_size比例大小，代表是batch_size的多少倍')
    parser.add_argument('--buffer_prob', type=float, default=10, help='buffer_size比例大小，代表是batch_size的多少倍')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--model_save_root', type=str,
                        default='/home/notebook/data/group/sihao_work/browse_recall/model_recall/weights/browse_search_query2search/',
                        help='模型保存路径')
    parser.add_argument('--exp_id', type=int, default=None)
    parser.add_argument('--metrics', type=str2list, default='hit,mrr,ndcg', help='')
    parser.add_argument('--topk_list', type=partial(str2list, trans_type="int"), default='5,10,50,100,200,300', help='')
    parser.add_argument('--run_eagerly', type=str2bool, default='false')
    parser.add_argument('--thread_num', type=int, default=4, help='读取数据线程数')
    parser.add_argument('--train_mode', type=str, default='test', choices=['train', 'test', 'online'], help='')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    args.prefetch_size = int(args.prefetch_prob * args.batch_size)
    args.buffer_size = int(args.buffer_prob * args.batch_size)
    print_args_table(args)

    is_debug = args.train_mode == "test"

    conf = Configuration(args.conf_path)
    labels = conf.features.labels
    # exp_conf = conf.networks if args.exp_id is None else conf.active_experiment(args.exp_id)

    print("Config Detail:")
    conf.print_features()

    data_dayno_conf = conf.get_conf_value("7days")
    model_name = conf.get_conf_value("class")
    dayno = conf.get_conf_value("dayno", str)
    loss = str2loss(conf.get_conf_value("loss"))
    valid_size = data_dayno_conf["valid"]

    train_dataset, valid_dataset = load_tfrecord_dayno_patten_split(
        args.train_tfr_root, dayno_patten=str2dayno(data_dayno_conf["train"]),
        feature_conf=conf, batch_size=args.batch_size, buffer_size=args.buffer_size, prefetch_buffer_size=args.prefetch_size,
        label_names=labels, thread_num=args.thread_num, shuffle=True, drop_remainder=True, sample_ratio=args.sample_ratio,
        valid_size=float(valid_size), is_debug=is_debug
    )

    eval_data_dict = get_dataset_dict_with_dayno(data_dayno_conf["eval"], args.eval_tfr_root, args.app_cand_tfr_root, args.smt_tfr_root,
                                                 conf, args.batch_size, labels, args.prefetch_size, args.buffer_size, drop_end=False,
                                                 thread_num=args.thread_num, is_debug=is_debug)

    strategy = auto_device_strategy()
    with strategy.scope():
        params = {"feature_conf": conf, "loss": loss, "name": model_name}
        model: Model = build_network(model_name, params)
        optimizer_main = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        model.compile(optimizer=optimizer_main, run_eagerly=args.run_eagerly)

    monitor = "val_auc"
    evaluator = Evaluator(args.task, eval_data_dict, args.topk_list, args.metrics, args.model_save_root, args.train_mode)
    early_stopping = EarlyStopping(monitor=monitor, patience=4, restore_best_weights=True)
    plateau = ReduceLROnPlateau(monitor=monitor, verbose=1, factor=0.5, patience=3)
    model.fit(train_dataset, epochs=args.epochs, verbose=1, callbacks=[evaluator, early_stopping, plateau], validation_data=valid_dataset)

    if args.train_mode == "online":
        best_model_save_path = os.path.join(args.model_save_root, "best_model", f"best_{args.task}_model.weights")
        model.save_weights(best_model_save_path)
        print(f"[SUCCESS] Best Model Has Been Saved At: {best_model_save_path}")
    else:
        print(f"[WARNING] Best Model Dose Not Save In Online Path.")
