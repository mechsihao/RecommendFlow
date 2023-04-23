# RecommendFlow (RecFlow)
## 项目特点
- 1.工业级的推荐模型训练框架，快速的数据IO、特征ETL、多卡训练支持
- 2.可直接从HDFS读取数据进行训练，可自动将数据转换为TFRecord加速模型训练
- 3.配置文件式读取设置字段和设置参数，让算法工程师专注于模型的开发，特征的ETL交给RecommendFlow吧


#### RecommendFlow (RecFlow) is a Industrialization recommendation system which can easy to build pipeline between local/hdfs train data and tensorflow2.x model.

#### RecommendFlow（RecFlow）是一个工业化推荐系统框架。它可以轻松的搭建数据pipeline（支持从HDFS或者本地读取），并且可以高效进行特征的读写、转换、喂入模型等操作。是数据和模型之间的高效桥梁。它致力于让算法工程师可以专注于模型的设计和思考中，至于数据的读取和特征转换（如哈希、查表、分箱等操作）的方法及其速度和效率等事情就交给RecFlow来操心吧。
