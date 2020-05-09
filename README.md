# TAAC2020
腾讯广告算法大赛2020

# TODO

- [x] LightGBM+voting
- [ ] LightGBM+LightGBM
- [ ] LightGBM+RNN
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

# 数据

给定的三个训练数据文件`user.csv ad.csv click_log.csv`的文件和外键关系如下：

![](img/TAAC2020.png)

# 使用
- `process_data.ipynb`将三个文件按照外键合并成一个文件，把`process_data.ipynb`中的数据路径修改一下即可。

合并之后的数据：
![](img/data_merged.png)

# 训练记录

LightGBM的baseline训练10个epoch，官网提交准确率0.91