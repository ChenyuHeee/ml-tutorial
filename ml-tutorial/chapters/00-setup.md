# 第 0 章：环境准备与工作方式（30–60 分钟）

目标：把机器学习学习所需的环境、目录结构和最小工程习惯建立起来。

本章的验收标准只有一个：**你能在 VS Code / 终端 / Jupyter 里使用同一个 Python 环境**（不再出现“notebook 能 import，终端不能”的情况）。

## 0.1 你要用到什么
- Python 3.11（建议）
- Jupyter（探索）
- scikit-learn（经典 ML）
- （后面）PyTorch（深度学习）

## 0.2 在 macOS 上创建环境（推荐 conda）
### 方案 A：conda（推荐）
1. 创建环境：
   - `conda env create -f ml-tutorial/env/environment.yml`
2. 激活环境：
   - `conda activate ml-tutorial`
3. 注册 Jupyter 内核：
   - `python -m ipykernel install --user --name ml-tutorial --display-name "Python (ml-tutorial)"`

4. 启动 Jupyter（任选一种）：
   - `jupyter lab`
   - 或 `jupyter notebook`

创建完成后，先做一次“你真的在这个环境里吗？”自检：
- `which python`
- `python -V`
- `python -m pip -V`
- `python -c "import sys; print(sys.executable)"`
- `python -c "import numpy, pandas, sklearn; print('OK', sklearn.__version__)"`

### 方案 B：venv + pip（也可以）
1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install -r ml-tutorial/env/requirements.txt`
4. `python -m ipykernel install --user --name ml-tutorial --display-name "Python (ml-tutorial)"`

5. 启动 Jupyter：
   - `jupyter lab`

> 如果你还没装 conda：你可以先用 venv 跑通，后续需要时再迁移。

如果你用的是 venv，建议在 VS Code 中把 `.venv/` 放在工作区根目录（不要散落到子目录），这样解释器识别更稳定。

## 0.3 建议的练习目录结构（你照着建即可）
在工作区根目录创建：
- `ml-work/`
  - `ch01-python-data/`
  - `ch02-ml-basics/`
  - `ch03-linear-regression/`
  - ...

每个章节目录里建议放：
- `notebook.ipynb`（探索）
- `train.py`（训练脚本，后续章节会教你写）
- `README.md`（本章结论与复盘）

## 0.4 最小工程习惯（现在就开始）
- 每次实验都记录：数据版本、随机种子、指标
- 不把“手动在 notebook 里点出来的结果”当最终结果：要脚本化
- 遇到 bug 的三件套：打印 `shape`、`dtype`、`head()`

建议你从现在就统一一个“可复现习惯”：
- 每个章节至少交付 1 个可运行脚本（例如 `train.py`）
- README 里必须写：如何安装、如何运行、输出了什么指标

你后面会越来越依赖这个习惯：
- Notebook 用来探索（画图、试想法）
- 脚本用来复现（别人/未来的你，一行命令跑出同样结果）

## 0.5 验收任务（必须完成）
1. 能打开 Jupyter，并选择内核 `Python (ml-tutorial)`
2. 新建一个 notebook，运行：
   - `import numpy as np, pandas as pd, sklearn`
3. 在终端运行：
   - `python -c "import sklearn; print(sklearn.__version__)"`

你应当看到类似输出（版本号可能不同）：
- `1.5.x` 或 `1.4.x`

## 0.6 VS Code 里你必须做的两件事（避免后面踩坑）
1. 选对 Python 解释器：
   - VS Code → Command Palette → “Python: Select Interpreter” → 选择 `ml-tutorial`（conda）或 `.venv`
2. 选对 Notebook 内核：
   - 打开 notebook → 右上角 Kernel → 选择 `Python (ml-tutorial)`

额外建议（非常实用）：
- 打开 VS Code 的 Terminal 后，先执行一次 `python -c "import sys; print(sys.executable)"`，确认它指向你的 conda/venv。
- 如果你经常“选错解释器”，可以在工作区设置里固定解释器路径（后面你需要我也可以帮你配）。

## 0.7 常见问题（按症状排查）
- `conda: command not found`：说明没装 conda 或 PATH 没生效；先用 venv 方案跑通。
- notebook 里 import 成功，但终端 import 失败：VS Code 解释器没选对；回到 0.6。
- `No module named sklearn`：依赖没装到当前环境；确认你已激活环境再 `pip install -r ...`。

更具体一点：
- **你以为激活了 conda，但其实没有**：在终端里跑 `conda info --envs` 看当前 `*` 在哪。
- **Kernel 列表里找不到 `Python (ml-tutorial)`**：重跑内核注册命令：`python -m ipykernel install --user --name ml-tutorial --display-name "Python (ml-tutorial)"`。
- **Jupyter 能跑，VS Code 运行脚本报错**：通常是 VS Code 的解释器不是同一个；用 0.2 的自检命令对比 `sys.executable`。

完成后，进入第 1 章。
