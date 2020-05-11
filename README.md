# TAAC2020
腾讯广告算法大赛2020

# TODO

- [ ] 传统机器学习如随机森林、决策树、SVM、朴素贝叶斯、贝叶斯网络、逻辑回归、AdaBoost等
- [ ] 全连接网络
- [x] LightGBM+Voting
- [x] ~~LightGBM+LightGBM~~(not work)
- [x] ~~LightGBM+RNN~~(not work)
- [ ] DeepFM、DeepFFM
- [ ] RNN
- [ ] GNN
- [ ] 集成学习

# 代码介绍

```bash
.
├── LICENSE
├── LightGMB.py         # LightGBM baseline
├── README.md
├── img
├── data                # 训练和测试数据
├── process_data.ipynb  # 将训练集ad.csv、user.csv合并到click_log.csv，测试集中的ad.csv合并到click_log.csv
└── tmp
```

# 数据探索

给定的三个训练数据文件`user.csv ad.csv click_log.csv`的文件和外键关系如下：
- `ad.csv`中一个素材id只能对应一个广告id，一个广告id对应多个素材id

![](img/TAAC2020.png)

测试集和训练集中ad.csv中：相同的ID占比：
```python
len(set(ad_test.advertiser_id.values.tolist()) & set(ad_train.advertiser_id.values.tolist()))/len(set(ad_test.advertiser_id.values.tolist()) | set(ad_train.advertiser_id.values.tolist()))
```
- `product_id`: 0.73
- `creative_id`: 0.49
- `product_category`: 1.0
- `advertiser_id`: 0.81
- `industry`: 0.96

# 获得训练数据
- `process_data.ipynb`将三个文件按照外键合并成一个文件，把`process_data.ipynb`中的数据路径修改一下即可。

合并之后的数据：
![](img/data_merged.png)

# 训练记录

- age和gender是随机的，提交准确率0.66
- age和gender都为1，提交准确率0.707
- LightGBM的baseline训练10和50个epoch，提交准确率均为0.91，其中age准确率0.22582、gender准确率0.687186。