# 模型语义召回工程

### 远程开发小技巧：
用以下这段代码来实现ssh和主机之间的接口映射，达到远程使用jupyter的效果
```shell
ssh -N -f -L localhost:8888:localhost:1818 oppoer@remote-dev.starfire.wanyol.com -p12548
```
然后连接 [http://localhost:8888/proxy/notebook/task-20230116120935-78384/lab]()


### 一、该工程的目的
- 第一阶段还是单纯的语义相关性阶段，词库召回的基础已打好，但是相关性还亟待优化
- 模型召回需要做到兼顾相关性和ecpm
  - 在迭代完纯语义之后，就需要将纯语义embedding作为特征，然后引入个性化特征来作向量召回了

### 二、该工程着手角度
纯语义阶段，有以下几条着手点：
    
- 1.特征式：
  - 1.1 预训练式 
    - [ ] 单纯利用曝光数据作为正样本，按照simbert的训练方式在simbert基础上继续训练一个权重，直接利用该权重embedding/whitening + embedding
    - [ ] 将正样本负样本均加入进来，在原始simbert基础上做nsp任务进行预训练，最后得到一个权重，直接利用该权重embedding/whitening + embedding
    - [ ] 在原simbert基础上加入 ltr loss
    - 这种办法比较难用评估集评估：
      - 可以将 app_name 用 index 存储起来，然后输入query检索top的召回率来评估 recall@topK
      - 也可以将 query向量 和 app_name向量 whitening后点积，评估 recall@precision [目前使用的是该方法]
  - 1.2 相似度训练式
    - [X] sentence-bert式训练，利用sentence-bert encode向量然后检索，评估时模型输出分数预测0/1
    - [X] cosent-bert式训练，同上
    - [ ] 三分类训练，同样使用sentence-bert/cosent，但是训练样本修改为 0,1,2的label，0为随机采样，1为曝光，2为点击，利用nli方式训练
    - [X] 三元组loss，训练
- 2.交互式 ：
  - 2.1 将两句话拼成一句预测是否曝光
    - [X] 利用roberta进行 效果较差
    - [ ] 利用simbert进行

### 三、该工程的目前准备和计划TODO
- 1.目前准备
  - simbert 
  - s-bert
  - cosent
  - bert-esim
  - interact
  - dssm
  - ltr模型
- 2.模型优化细节

### 四、其他优化点
#### 1.模型能力是召回的其中一点但是数据更是其中重要的一环。这里的数据分为两部分
- 训练数据及验证数据
- 预测数据，理想情况下，你需要预测明天的所有query，并且将其用离线的方式预测好存储

### TODO
- 1.莫比乌斯代码实现
- 2.que2search代码完全体实现(包含doc塔的多标签分类任务)
- 3.正负样本加权代码完全实现(正样本`根号 点击次数`、负样本`log query请求次数`)
- 4.构造辅助塔中正样本之间的pair-wise的bid比较，将cosent思想用在辅助塔的bid对比上
- 5.完善加入bid后的辅助评估指标（如bid分桶auc、bid@5等指标）