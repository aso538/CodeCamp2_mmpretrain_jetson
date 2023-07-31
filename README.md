# 分类模型微调以及在Jetson平台上进行测速

# 1 项目介绍
本项目使用mmpretrain对预训练的分类模型在垃圾分类数据集上进行微调，同时对微调后模型进行测试、部署、测速。

本项目在Geforce RTX2060进行训练微调，所使用模型为[efficientnet-b1](mmpretrain/configs/efficientnet/README.md)，
同时我们选择官方所提供的预训练模型作为base模型在其基础上进行微调。
- [任务详情](https://github.com/open-mmlab/OpenMMLabCamp/discussions/566)
- base模型[Config](./mmpretrain/configs/efficientnet/efficientnet-b1_8xb32_in1k.py)和[权重](https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b1_3rdparty-ra-noisystudent_in1k_20221103-756bcbc0.pth)
- 微调后模型[Config](./mmpretrain/configs/efficientnet/efficientnet-b1_ingarbage.py)和[权重]()

# 2 训练/测试环境



# 3 数据集和数据处理

本项目选择的数据集为[garbage数据集](https://aistudio.baidu.com/aistudio/datasetdetail/77996)

![image_data](./1121.jpg)


# 4 模型训练与功能测试

## 4.1 模型训练

微调参数如下：

| 参数名        |       |
|------------|-------|
| max_epochs | 10    |
| batch_size | 16    |
| lr         | 0.01  |
| milestones | 2,5,8 |


训练命令为：

```
python tools/train.py ./configs/efficientnet/efficientnet-b1_ingarbage.py
```

## 4.2 功能测试

功能测试命令为：

```
# Finetuned model test.
cd mmpretrain
python tools/test.py ./work_dir/garbage/efficientnet-b1_ingarbage.py ./work_dir/garbage/epoch_7.pth

```

微调后的模型功能测试结果为：

```
accuracy/top1: 88.9614
```

- 本次训练日志：[点击](./mmpretrain/work_dir/garbage/20230727_223655/20230727_223655.log)
- 训练后的模型本次功能测试结果：[点击](./mmpretrain/work_dir/garbage/20230729_160930/20230729_160930.json)

