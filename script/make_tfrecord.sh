#!/bin/bash
# 功能描述：
# 创建者：李思浩（80302421）
# 创建日期：20220607

if [ $# -eq 0 ];then
  echo "Online Mode..."
elif [[ $# -eq 1 ]] ;then
  if [[ $1 == "t" ]] ;then
    echo "Debug Mode..."
  else
    echo "Online Mode..."
  fi
fi

FEATURE_CONF="/home/notebook/data/group/sihao_work/browse_recall/model_recall/conf/market_feature_conf.json"
FILE_ROOT="/home/notebook/data/group/sihao_work/browse_recall/model_recall/data/market_data/train_test_data/20221104"
FILE_LIST="*q2s.csv eval_q2s.*.csv app_cand_q2s_fea.csv"  # 支持google glob通配符 支持HDFS文件


function make_tfrecord_data()
{
  src_file=$1
  out_dir=$2
  python /home/notebook/data/group/sihao_work/browse_recall/model_recall/utils/make_tfrecord.py \
        "$FEATURE_CONF" "$src_file" "$out_dir"
  if [ $? -eq 0 ]; then
    echo "Tfrecord转换成功"
  else
    echo "Tfrecord转换失败，请检查！"
  fi
}

FILE_ARRAY=($FILE_LIST)


for FILE in ${FILE_ARRAY[@]}
do
    echo "开始转换文件：$FILE_ROOT/$FILE"
    make_tfrecord_data "$FILE_ROOT/$FILE" "$FILE_ROOT"
done