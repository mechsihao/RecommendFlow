# 配置文件说明
- 1.配置文件字符串中的$xxx均会被该配置文件中的变量替代，因此可以很灵活的运用'$'简化配置文件
- 2.$key 会递归的寻找配置中第一个出现的 key 所对应的 value
- 3.必须存在顶层Key为：Features
- 4.Networks、Experiments这两字段代表网络的结构和实验配置，可有可无，但是意义明确
- 5.其余字段均可以按需自定义，无限制
# 字段说明：
## Features：
- 1.Features字段和格式基本固定，可新增字段但不可删除字段，若feature_group里面为空，则需要配置为 `feature_group: {}` 或 `feature_group:`
- 2.feature_group:
  - pip_rec的特征配置是分组的，你可以将相同配置/相同意义/相同类型的特征配置到一个特征组内。
  - 如果在features中配置的特征组在feature_group中找不到，那么将会单独成组。
- 3.feature_fields:
  - `feature_fields` 字段固定为field, type, tower, deal, vocab, embedding_dim, pooling, working
  - *field* 代表特征组，若*field*不在*feature_group*中，则 *field* 会单独为一个feature name
  - *type* 目前仅支持三种：int、float、str，注意：pip_rec的输入均为list类型，即 int_list、float_list、str_list，type指的是list中的元素的数据类型
  - *tower* 代表特征所在塔，目前仅支持null、user、ad、context、label，如果是ctr模型则不需要这个字段，可以设置为一个统一的值，如null
  - *deal* 代表特征处理方式，目前支持的类型为：
    - null 不处理，原值输入
    - numeric 数值型，仅支持type float
    - discrete 分箱特征，需提供分箱边界list，仅支持type float
    - hashing 哈希特征，需提供 *hashing buck size*，仅支持type str
    - lookup 查表特征，需提供词表路径, 仅支持type为 int、str
    - image：图像序列化，目前只支持tensorflow自带的图像序列化和反序列化
    - embedding：embedding序列化，目前也只支持numpy和tf自带的序列化和反序列化
    - token_id：token_id是事先将其encode成id了
    - bert_encode：bert_encode输入是str，由框架将其encode成id
  - *vocab* 对于hashing类，即为hashing buck size，其余为词表地址，或者词表内容。
  - *embedding_dim* 代表输出的embedding大小，对于numeric类型不生效，提供-1即可
  - *pooling* 因为输入均为list，对于单值特征，list中仅有一个元素，但对于多值特征，list中可能有多个特征，pooling就是该list中特征的多值的embedding的pooling方式：
    - null：不pooling，有多少维出来多少维，这种最好需要不同样本特征的list重元素个数一致
    - avg：均值
    - min：最小值
    - max：最大值
    - sum：求和
    - cls：取第一个
  - *working* 该特征是否生效，true生效，false不生效
  - Features部分配置示例
```yaml
Features:
  feature_group:
    query: [query_tok_id, query_seg_id]
    query_sug: [query_sug_tok_id, query_sug_seg_id]
    query_nlp_token: [query_2gram, query_3gram, query_token]
    app_name: [app_name_tok_id, app_name_seg_id]
    app_desc: [app_desc_tok_id, app_desc_seg_id]
    app_nlp_token: [app_kws, app_name_2gram, app_name_3gram, app_name_token, app_desc_token]
    exp_disc: [app_all_exp, app_avg_exp]
    down_disc: [app_all_down, app_avg_down]
    income_disc: [app_all_income, app_avg_income]
  feature_fields: [group, type, tower, deal, vocab, embedding_dim, pooling, working]
  features:
    query,int,user,token_id,null,-1,cls,true
    query_sug,int,user,token_id,null,-1,cls,false
    sc,str,user,null,null,-1,null,false
    query_nlp_token,str,user,hashing,5000,16,sum,false
    clk_app_ids,str,user,hashing,3000,16,sum,false
    clk_app_kws,str,user,hashing,5000,16,sum,false
    clk_ad_2cat_ids,int,user,lookup,$ad_cat2,16,sum,false
    clk_ad_3cat_ids,int,user,lookup,$ad_cat3,16,sum,false
    clk_shop_2cat_ids,int,user,lookup,$shop_cat2,16,sum,false
    clk_shop_3cat_ids,int,user,lookup,$shop_cat3,16,sum,false
    app_name,int,ad,token_id,null,-1,cls,true
    app_desc,int,ad,token_id,null,-1,cls,false
    app_id,str,ad,hashing,3000,16,sum,true
    app_nlp_token,str,ad,hashing,5000,16,sum,false
    developer_id,str,ad,hashing,1000,16,sum,false
    app_shop_2cat_id,int,ad,lookup,$shop_cat2,16,sum,false
    app_shop_3cat_id,int,ad,lookup,$shop_cat3,16,sum,false
    app_ad_2cat_id,int,ad,lookup,$ad_cat2,16,sum,false
    app_ad_3cat_id,int,ad,lookup,$ad_cat3,16,sum,false
    top_cat,str,ad,lookup,$top_cat,16,sum,false
    app_qv,float,ad,discrete,$qv_disc,16,sum,false
    exp_disc,float,ad,discrete,$exp_disc,16,sum,false
    down_disc,float,ad,discrete,$down_disc,16,sum,false
    income_disc,float,ad,discrete,$income_disc,16,sum,false
    app_avg_ctr,float,ad,discrete,$ctr_disc,16,sum,false
    label,float,label,numeric,null,-1,null,true
    down,float,null,numeric,null,-1,null,true
```
## Experiments：
- 其中experiment_fields和experiments字段是必须的
- experiment_fields中有一个特殊字段features，如果配置该字段代表对特征的增减实验
  - '-'代表哦减去某个特征，'+'代表增加某个特征
  - 注意，这里的特征必须在前面的Features中被配置
  - Features配置文件示例
```yaml
Experiments:
  feature_exp:
    no_del: []
    del_sug_and_desc: [-query_sug_tok_id, -query_sug_seg_id, -app_desc_tok_id, -app_desc_seg_id]
  experiment_fields: [exp_id, loss, train_data, dayno_conf, features]
  experiments:
    0,cosent,$train_data1,$7days,$del_sug_and_desc
    1,cosent,$train_data2,$7days,$del_sug_and_desc
    2,cosent,$train_data2,$30days,$del_sug_and_desc
    3,cosent,$train_data3,$7days,$del_sug_and_desc
    4,cosent,$train_data3,$30days,$del_sug_and_desc
    3,cosent,$train_data3,$7days,$no_del
    4,cosent,$train_data3,$30days,$no_del
```