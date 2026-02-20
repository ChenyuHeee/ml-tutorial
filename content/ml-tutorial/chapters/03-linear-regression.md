# 第 3 章：线性回归与第一个可复现训练脚本（1 周）

目标：你能把一个回归任务做成**一键可运行**的 `train.py`，并输出稳定的 MAE/RMSE 结果（别人拿到你的代码也能复现）。

> 本章使用无需注册的数据集：`sklearn.datasets.fetch_california_housing`。

## 3.1 概念最小集（够用就行）
- 线性模型：$\hat{y} = w^T x + b$
- 损失：MSE（均方误差）
- 指标：
  - MAE：更直观、对异常值更稳
  - RMSE：更惩罚大误差（更怕“错得离谱”）
- 正则化（知道即可）：L2（岭回归）抑制过大权重，通常更稳

## 3.2 本章你要交付的东西（先把目标钉死）
在你的练习目录（推荐：`ml-work/ch03-linear-regression/`）里，至少有：
- `train.py`：训练 + 评估 + 保存模型（一个命令跑完）
- `evaluate.py`：加载模型 + 在测试集输出指标
- `README.md`：复现命令 + 你得到的指标

如果你只看网页也能做：看到“把下面代码保存为 …”就新建文件，把代码块粘进去即可。

## 3.3 直接可用：`train.py`（复制即可运行）
把下面代码保存为 `train.py`，然后运行：

```bash
python train.py
```

```python
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  return float(mean_squared_error(y_true, y_pred, squared=False))


def main() -> None:
  # 1) 读取数据（无需下载文件）
  ds = fetch_california_housing(as_frame=True)
  X = ds.data
  y = ds.target

  # 2) 切分
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
  )

  # 3) 预处理 + 模型（先用最简单的数值缩放即可）
  pipe = Pipeline(
    steps=[
      ("scaler", StandardScaler()),
      ("model", Ridge(alpha=1.0, random_state=42)),
    ]
  )

  # 4) 训练
  pipe.fit(X_train, y_train)

  # 5) 评估
  pred = pipe.predict(X_test)
  mae = mean_absolute_error(y_test, pred)
  r = rmse(y_test.to_numpy(), pred)

  print("dataset: california_housing")
  print("n_train:", len(X_train), "n_test:", len(X_test))
  print("mae:", round(float(mae), 4))
  print("rmse:", round(float(r), 4))

  # 6) 保存（把预处理和模型一起存）
  Path("models").mkdir(exist_ok=True)
  out_path = Path("models") / "ridge.joblib"
  joblib.dump(pipe, out_path)
  print("saved:", out_path.as_posix())


if __name__ == "__main__":
  main()
```

你应当能看到类似输出（数值会略有差异，但结构应该一致）：

```text
dataset: california_housing
n_train: 16512 n_test: 4128
mae: ...
rmse: ...
saved: models/ridge.joblib
```

## 3.4 直接可用：`evaluate.py`（加载模型再评估一次）
把下面保存为 `evaluate.py`，运行：

```bash
python evaluate.py
```

```python
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  return float(mean_squared_error(y_true, y_pred, squared=False))


def main() -> None:
  model_path = Path("models") / "ridge.joblib"
  pipe = joblib.load(model_path)

  ds = fetch_california_housing(as_frame=True)
  X = ds.data
  y = ds.target
  _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  pred = pipe.predict(X_test)
  print("loaded:", model_path.as_posix())
  print("mae:", round(float(mean_absolute_error(y_test, pred)), 4))
  print("rmse:", round(float(rmse(y_test.to_numpy(), pred)), 4))


if __name__ == "__main__":
  main()
```

## 3.5 验收任务（必须完成）
- `python train.py` 能一键输出 MAE/RMSE，并保存 `models/ridge.joblib`
- `python evaluate.py` 能加载模型并输出同样口径的 MAE/RMSE
- `README.md` 至少写清楚三件事：
  - 你用的数据集是什么
  - 训练/评估命令是什么
  - 你跑出来的指标是多少

## 3.6 常见坑（你遇到就回来看）
- 没固定 `random_state`：每次切分不同，指标不同
- 把指标算错：RMSE 不是 MSE；别把 `squared=False` 忘了
- 保存了“纯模型”没保存预处理：后面推理时数据分布不一致会崩（所以我们存 `Pipeline`）

完成本章后进入第 4 章（分类 + 概率 + 阈值）。
