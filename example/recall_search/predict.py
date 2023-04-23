import argparse
from functools import partial

import tensorflow as tf

from config_parser.configuration import Configuration
from utils.str_parser import str2dict, str2loss, str2list, str2bool
from backend.core.dataloader import load_multi_tfrecord_dayno_patten
from backend.utils.model_utils import build_network
from utils.print_util import print_args_table
from tensorflow.keras.models import Model


def arg_parser():
    parser = argparse.ArgumentParser(description='BertDssm Train')
    parser.add_argument('--task', type=str, default='browse_search')
    parser.add_argument('--model', type=str, default='siamese_bert.BertModel')
    parser.add_argument('--conf_path', type=str,
                        default='/home/notebook/data/group/sihao_work/browse_recall/model_recall/conf/market_dssm_bert_conf.json',
                        help='预训练模型加载路径')
    parser.add_argument('--infer_tfr_root', type=str,
                        default='hdfs://ad-hdfs/hive-dw/ad/tag/recall_search/que2search/browse_search/daily_train/YYYYMMDD/train_tfr_neg_10_exp5_ctr_002',
                        help='模型训练数据路径')
    parser.add_argument('--app_cand_tfr_root', type=str,
                        default='hdfs://ad-hdfs/hive-dw/ad/tag/recall_search/que2search/browse_search/daily_app_cand/YYYYMMDD/app_cand_tfr',
                        help='app数据路径')
    parser.add_argument('--dayno', type=str2dict, default='train=20221217-7;valid=0.1,20221217;eval=20221217+:1', help='数据日期配置')
    parser.add_argument('--sample_ratio', type=float, default='1', help='数据采样比例')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=768, help='batch大小')
    parser.add_argument('--prefetch_prob', type=float, default=2, help='prefetch_size比例大小，代表是batch_size的多少倍')
    parser.add_argument('--buffer_prob', type=float, default=2, help='buffer_size比例大小，代表是batch_size的多少倍')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--model_save_root', type=str,
                        default='/home/notebook/data/group/sihao_work/browse_recall/model_recall/weights/browse_search_query2search/20221120',
                        help='模型保存路径')
    parser.add_argument('--loss', type=partial(str2loss, patten="core"), default='mse')
    parser.add_argument('--gpus', type=str2list, default='GPU:0,GPU:1', help='指定GPU训练')
    parser.add_argument('--online_model_save_root', type=str, default='', help='最佳encoder保存路径')
    parser.add_argument('--metrics', type=str2list, default='hit,mrr,ndcg', help='')
    parser.add_argument('--topk_list', type=partial(str2list, trans_type="int"), default='5,10,50,100,200,300', help='')
    parser.add_argument('--core_update_metrics', type=str2dict, default='hit@50=[-0.1,inf];auc=[-0.1,inf]', help='监控上线的指标及允许波动的范围')
    parser.add_argument('--run_eagerly', type=str2bool, default='true')
    parser.add_argument('--thread_num', type=int, default=4, help='读取数据线程数')
    parser.add_argument('--train_mode', type=str, default='test', choices=['train', 'test', 'online'], help='')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    args.prefetch_size = int(args.prefetch_prob * args.batch_size)
    args.buffer_size = int(args.buffer_prob * args.batch_size)
    print_args_table(args)

    is_debug = args.train_mode == "test"

    features_conf = Configuration(args.conf_path)
    labels = features_conf.features.labels

    print("Config Detail:")
    features_conf.print_features()

    train_dataset = load_multi_tfrecord_dayno_patten(
        args.infer_tfr_root, dayno_patten=args.dayno, feature_conf=features_conf,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        prefetch_buffer_size=args.prefetch_size,
        label_names=labels,
        thread_num=args.thread_num,
        shuffle=True,
        drop_remainder=True,
        sample_ratio=args.sample_ratio,
        is_debug=is_debug
    )

    # 单机多卡策略, 测试环境只用单卡
    strategy = tf.distribute.MirroredStrategy([args.gpus[0]] if is_debug else args.gpus)

    with strategy.scope():
        params = {"feature_conf": features_conf, "loss": args.loss, "name": args.model}
        model: Model = build_network(args.model, params, model_checkpoint=args.load_model_path)
        optimizer_main = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        model.compile(optimizer=optimizer_main, run_eagerly=args.run_eagerly)

    model.predict(train_dataset)
