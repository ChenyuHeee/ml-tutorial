# 开始之前（必须读一遍）

如果你是从零开始，建议按这个顺序推进，确保不会卡在环境或概念上。

> 重要：你现在看到的是网页版本。读者不需要有任何“仓库路径/我的本地文件”，也能跟着做。

## 0) 只走一条路（推荐）：下载仓库 + venv + pip
你只需要照着下面做，**不要做任何“二选一”**。这条路在 macOS 上最稳定。

### Step 1：拿到教程文件（两种都行，选你更顺手的）
- 有 Git：

```bash
git clone https://github.com/ChenyuHeee/ml-tutorial.git
cd ml-tutorial
```

- 没有 Git：在仓库页面点 **Code → Download ZIP**，解压后进入该文件夹（核心是进入项目根目录）。

### Step 2：创建并激活虚拟环境（venv）
在项目根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3：安装依赖

```bash
pip install -r env/requirements.txt
python -m ipykernel install --user --name ml-tutorial --display-name "Python (ml-tutorial)"
```

### Step 4：用 3 条命令自检（通过就算起步成功）

```bash
which python
python -c "import sys; print(sys.executable)"
python -c "import numpy, pandas, sklearn; print('OK', sklearn.__version__)"
```

如果上述三条都 OK，你就已经“起步成功”，直接进入第 0 章的验收任务。

> 想只看网页不下载仓库也可以，但会更麻烦（需要自己建文件夹/粘代码/装依赖）。先按这条主线跑通，再考虑“纯网页复制”。

## 1) 然后按第 0 章再过一遍（10 分钟）
- 打开 [chapters/00-setup.md](chapters/00-setup.md)
- 只做“验收任务”，确保 VS Code/终端/Jupyter 使用同一个环境

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
