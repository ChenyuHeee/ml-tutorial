# 开始之前（必须读一遍）

如果你是从零开始，建议按这个顺序推进，确保不会卡在环境或概念上。

> 重要：你现在看到的是网页版本。读者**不需要**有我本地的工作区，也能跟着做。

## 0) 你有两种跟学方式（二选一）

### 方式 A：获取仓库（推荐，最省心）
如果你电脑里有 Git：

```bash
git clone https://github.com/ChenyuHeee/ml-tutorial.git
cd ml-tutorial
```

然后用 VS Code 打开这个文件夹，再按第 0 章创建环境。

如果你不想用 Git：在仓库页面点 **Code → Download ZIP**，解压后用 VS Code 打开。

### 方式 B：只看网页也能做（复制代码块）
你可以在任意文件夹新建一个练习目录（例如 `ml-work/`），然后：
- 看到“把下面代码保存为 xxx.py”时：你就新建文件，把网页里的代码块粘进去
- 看到“在终端运行 xxx”时：你就照抄命令运行

本教程会尽量把关键代码都写成代码块，方便你直接复制。

## 1) 先把环境跑通（30–60 分钟）
- 完成 [chapters/00-setup.md](chapters/00-setup.md)
- 验收：你能在终端运行 Python，并在 Jupyter 里 `import numpy, pandas, sklearn`（后面会装）

## 2) 第一周只做两件事
- 跟完 [chapters/01-python-data.md](chapters/01-python-data.md) 的必做练习
- 跟完 [chapters/02-ml-basics.md](chapters/02-ml-basics.md) 的验收任务

## 2.5) 随身携带的速查（强烈建议收藏）
- 指标与验证：见 [appendix/01-metrics-and-validation.md](appendix/01-metrics-and-validation.md)
- 数据泄漏排查：见 [appendix/02-leakage-checklist.md](appendix/02-leakage-checklist.md)
- sklearn / PyTorch 配方与排错：见 [appendix/README.md](appendix/README.md)

## 3) 遇到卡点的处理顺序
1. 把报错信息复制出来（不要只说“报错了”）
2. 打印关键变量：shape、dtype、head()
3. 简化到最小复现（最少代码 + 最小数据）
4. 仍卡住：把你做到了哪一步、错误信息、你猜原因发给我，我会按你当前状态继续指导
