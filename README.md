# TAAC2020
腾讯广告算法大赛2020

# TODO

- [ ] 传统机器学习如随机森林、决策树、SVM、朴素贝叶斯、贝叶斯网络、逻辑回归、AdaBoost等
- [ ] 全连接网络
- [x] LightGBM+Voting
- [x] ~~LightGBM+LightGBM~~(not work)
- [x] ~~LightGBM+RNN~~(not work)
- [ ] RNN
- [ ] TF-IDF
- [ ] DeepFM、DeepFFM
- [ ] GNN
- [ ] 集成学习

## 传统机器学习

随机森林：0.89

## 处理成序列问题

把每个点击的creative_id或者ad_id当作一个词，把一个人90天内点击的creative_id或者ad_id列表当作一个句子，使用word2vec来构造creative_id或者ad_id嵌入表示。最后进行简单的统计操作得到用户的向量表示。这种序列简单聚合导致信息损失，显得是非常的粗糙，需要进一步引入attention等方法。

上述方法可以直接使用传统的GBDT相关模型进行，1.3应该没问题。下面可以考虑序列建模方式。例如RNN/LSTM/GRU，这类方法将用户行为看做一个序列，套用NLP领域常用的RNN/LSTM/GRU方法来进行建模。

## TF-IDF

NLP中常用的做法，将用户点击序列中的creative_id或者ad_id集合看作一篇文档，将每个creative_id或者ad_id视为文档中的文字，然后使用tfidf。当然这也下来维度也非常高，可以通过参数调整来降低维度，比如sklearn中的TfidfVectorizer，可以使用max_df和min_df进行调整。

## GNN

将用户的访问记录看作图，利用图神经网络提取user_id、creative_id、ad_id等的Embedding，利用提取的Embedding输入下游模型，或者将访问记录看作序列输入序列模型。

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