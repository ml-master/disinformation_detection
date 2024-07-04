# 知识感知的提示学习假新闻检测论文复现
学号：2310274032 姓名：zc

这个仓库的代码为Fake news detection via knowledgeable prompt learning的复现代码

## 环境配置

- python 3.12
- pytorch 2.3.0
- cuda 12.1

## 数据处理

原始数据为MegaFake中的style-based fake和style-based legitimgate数据集，地址：https://github.com/SZULLM/MegaFake

首先需要对数据集进行预处理，原始数据集放在dataset目录下，预处理后的数据集放在processed_data目录下

运行以下代码，对数据集进行预处理

```
python data.py
```

## 模型训练

模型代码在model.py文件中，包括原论文中提出的模型以及另外两个对比模型

模型训练的配置文件在config目录下

运行以下代码，训练模型

```
python run.py
```

复现代码提供了不同模型以及few shot模型训练的代码

```
python run_few_shot.py
```

## 实验结果

对比实验结果如下：

其中KPL为论文中提出的模型，FT: fine tuing，PT: prompt tuning

| 方法                                      | F1     | acc    |
| ----------------------------------------- | ------ | ------ |
| KPL-full(Knowledge-based prompt learning) | 0.8715 | 0.8713 |
| KPL-2 shots                               | 0.6320 | 0.6213 |
| KPL-4 shots                               | 0.6797 | 0.6885 |
| KPL-8 shots                               | 0.6891 | 0.6873 |
| KPL-16 shots                              | 0.6900 | 0.6911 |
| KPL-100 shots                             | 0.7365 | 0.7435 |
| FT-2 shots(Fine Tune)                     | 0.3791 | 0.4087 |
| FT-4 shots                                | 0.5362 | 0.6689 |
| FT-8 shots                                | 0.5391 | 0.5402 |
| FT-16 shots                               | 0.6052 | 0.6562 |
| FT-100 shots                              | 0.6106 | 0.6899 |
| PT-2 shots(Prompt Tune)                   | 0.3547 | 0.4173 |
| PT-4 shots                                | 0.6413 | 0.6308 |
| PT-8 shots                                | 0.6849 | 0.6800 |
| PT-16 shots                               | 0.7393 | 0.7457 |
| PT-100 shots                              | 0.7408 | 0.7570 |

模型使用的评估指标包括准确率，F1分数、精度以及召回率

消融实验结果

| 模型     | KPL     |        |        |        |        |
| -------- | ------- | ------ | ------ | ------ | ------ |
| few shot | 2       | 4      | 8      | 16     | 100    |
| acc      | 0.6213  | 0.6885 | 0.6873 | 0.6911 | 0.7435 |
| 模型     | KPL -AW |        |        |        |        |
| acc      | 0.5984  | 0.6643 | 0.6785 | 0.6889 | 0.7362 |
| 模型     | KPL -LT |        |        |        |        |
| acc      | 0.5743  | 0.6483 | 0.6578 | 0.6673 | 0.7145 |
| 模型     | KPL -DT |        |        |        |        |
| acc      | 0.5702  | 0.6434 | 0.6478 | 0.6538 | 0.6965 |

在真新闻和假新闻混合数据集中的实验评估结果如下

| 数据集                          | f1     | acc    |
| ------------------------------- | ------ | ------ |
| style-based fake and legitimate | 0.8715 | 0.8713 |
| style-based fake                | 0.9308 | 0.8706 |
| style-based legitimate          | 0.9773 | 0.9557 |

Please cite our paper if it is helpful to your work.
```latex
@article{jiang-etal-fakenews,
  title = {Fake news detection via knowledgeable prompt learning},
  journal = {Information Processing & Management},
  volume = {59},
  number = {5},
  pages = {103029},
  year = {2022},
  issn = {0306-4573},
  doi = {https://doi.org/10.1016/j.ipm.2022.103029},
  url = {https://www.sciencedirect.com/science/article/pii/S030645732200139X},
  author = {Gongyao Jiang and Shuang Liu and Yu Zhao and Yueheng Sun and Meishan Zhang},
  keywords = {Fake news detection, Prompt learning, Pretrained language model, Knowledge utilization},
}
```
