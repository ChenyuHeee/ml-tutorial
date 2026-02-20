# 第 8 章：PyTorch 入门（1–2 周）

目标：你能用 PyTorch 训练一个小模型，掌握训练循环、GPU/CPU、过拟合诊断。

## 8.1 安装说明（重要）
PyTorch 的安装会因 CPU/GPU 与 CUDA 版本不同而不同：
- 请优先按官网安装命令安装
- 如果你先只做 CPU 也完全可以

## 8.2 概念最小集
- Tensor 与自动求导（autograd）
- Dataset/DataLoader
- 模型：`nn.Module`
- 损失与优化：`nn.CrossEntropyLoss`、`torch.optim.Adam`
- 训练循环：train/eval 模式切换

你必须掌握的训练循环最小清单：
- 选择 `device`（cpu / cuda / mps）并把模型与 batch 移过去
- `model.train()` 与 `model.eval()` 切换（评估时关掉 dropout 等）
- 用 `torch.no_grad()` 包住评估
- 每个 epoch 输出 loss/acc（至少在 train 与 val 各 1 组）

## 8.3 必做练习
- 用一个小数据集做分类（MNIST 或 CIFAR-10 都可以）
- 实现：
  - `train.py`：训练、打印 loss/acc
  - `evaluate.py`：在验证集评估
- 做 2 次对比实验：
  1) 不加正则（容易过拟合）
  2) 加 dropout 或 weight decay（看曲线变化）

建议你把每次实验的配置写到 README：
- 模型结构（层数、隐藏维度、是否 dropout）
- optimizer 与学习率
- batch size 与 epoch

## 8.4 验收任务
在 `ml-work/ch08-pytorch/` 里完成：
- 训练曲线截图或日志（至少 10 个 epoch）
- 写 5 行结论：你观察到的过拟合/欠拟合迹象是什么？你怎么改？

验收标准：
- 你能提供一段训练日志（或一张曲线图），看得出 train 与 val 的差距变化
- 你能解释：你改的正则/超参为什么可能有效

完成本章后进入最终项目。
