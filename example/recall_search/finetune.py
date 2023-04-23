import argparse
import os
from functools import partial
import tensorflow as tf
from backend.core.callbacks import ModelCheckpoint
from utils.str_parser import str2loss, str2list, str2bool
from backend.utils.model_utils import build_network
from backend.core.dataloader import load_tfrecord_dayno_patten
from config_parser.config_utils import load_config, print_conf
from utils.print_util import print_args_table
from utils.str_parser import str2dayno
from tensorflow.keras.models import Model


def arg_parser():
    parser = argparse.ArgumentParser(description='BertDssm Train')
    parser.add_argument('--task', type=str, default='browse_search')
    parser.add_argument('--model', type=str, default='siamese_bert.BertModel')
    parser.add_argument('--conf_path', type=str,
                        default='/home/notebook/data/group/sihao_work/browse_recall/model_recall/conf/market_dssm_bert_conf.json',
                        help='预训练模型加载路径')
    parser.add_argument('--train_tfr_root', type=str,
                        default='hdfs://ad-hdfs/hive-dw/ad/tag/recall_search/que2search/browse_search/daily_train/YYYYMMDD/train_tfr_neg_10_exp5_ctr_002',
                        help='模型训练数据路径')
    parser.add_argument('--dayno_pattern', type=str, default='20221217', help='数据日期配置')
    parser.add_argument('--model_save_root', type=str,
                        default='/home/notebook/data/group/sihao_work/browse_recall/model_recall/weights/browse_search_query2search')
    parser.add_argument('--sample_ratio', type=float, default='1', help='数据采样比例')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch大小')
    parser.add_argument('--prefetch_prob', type=float, default=10, help='prefetch_size比例大小，代表是batch_size的多少倍')
    parser.add_argument('--buffer_prob', type=float, default=10, help='buffer_size比例大小，代表是batch_size的多少倍')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数, 日度finetune的epoch一般不需要很多')
    parser.add_argument('--loss', type=partial(str2loss, patten="core"), default='mse')
    parser.add_argument('--gpus', type=str2list, default='GPU:0,GPU:1', help='指定GPU训练')
    parser.add_argument('--run_eagerly', type=str2bool, default='false')
    parser.add_argument('--thread_num', type=int, default=4, help='读取数据线程数')
    parser.add_argument('--train_mode', type=str, default='train', choices=['train', 'test'], help='')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    args.prefetch_size = int(args.prefetch_prob * args.batch_size)
    args.buffer_size = int(args.buffer_prob * args.batch_size)
    print_args_table(args)

    is_debug = args.train_mode == "test"

    features_conf = load_config(args.features_conf_path)
    labels = features_conf["labels"]["cols"]

    print("Config Detail:")
    print_conf(args.features_conf_path)
    dayno = max(str2dayno(args.dayno_pattern, mode="list"))

    train_dataset = load_tfrecord_dayno_patten(
        args.train_tfr_root,
        dayno_patten=str2dayno(args.dayno_pattern),
        feature_conf=features_conf,
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

    ckpt = ModelCheckpoint(args.model_save_root, "model.weights")
    model.fit(train_dataset, epochs=args.epochs, verbose=1, callbacks=[ckpt])

    model.save_weights(os.path.join(args.online_model_save_root, "best_model", f"best_{args.task}_model.weights"))
    print("-END-")
