# 第 5 章：特征工程与数据泄漏（sklearn Pipeline）（1–2 周）

目标：你能用 Pipeline 把预处理与模型训练“绑定在一起”，彻底避免常见泄漏。

## 5.1 概念最小集
- 数据泄漏（data leakage）：训练时看到了不该看的信息
- 正确做法：所有从数据学出来的变换（填充、缩放、编码）必须只在训练集上 fit
- Pipeline 的价值：把 fit/transform 的边界固化

## 5.2 你要掌握的 sklearn 组件
- `train_test_split`
- `ColumnTransformer`
- `SimpleImputer`（缺失值）
- `OneHotEncoder`（类别编码）
- `StandardScaler`（数值缩放）
- `Pipeline`

你最终要长成的结构（先看懂，再照着写）：
- `preprocess = ColumnTransformer([...])`
- `model = LogisticRegression(...)`（或回归模型）
- `clf = Pipeline([('preprocess', preprocess), ('model', model)])`

这样做的好处：任何 `fit` 都会先在训练集上 fit 预处理器；你不会“不小心”用全量数据统计均值/方差。

## 5.3 必做练习
- 用你第 4 章的数据，把预处理全部迁移到 Pipeline
- 对比：
  - 不用 Pipeline（容易泄漏）
  - 用 Pipeline（正确）
- 你要输出：同样的验证方式下，两者指标是否变化？为什么？

你要在 README 里回答这两个问题：
1. 你之前的写法哪里可能发生泄漏？（哪一步在全量数据上学了统计量？）
2. Pipeline 写法如何强制你在训练集上 fit？

## 5.4 验收任务
在 `ml-work/ch05-pipeline/` 里完成：
- `train.py`：使用 Pipeline 训练并输出指标
- `README.md`：列出数值/类别特征列表，并解释你的缺失值策略

---

下面把本章写“写死”：你只要复制代码块、运行、对照输出，就能完成。

## 5.5 错误写法（不要这么做）：在切分前 fit 预处理
这类代码的特征是：`fit_transform` 在 **全量数据** 上发生。

```python
# 伪代码示例（不要抄到你的项目里）
X_all = preprocess.fit_transform(X_all)
X_train, X_test = train_test_split(X_all, ...)
model.fit(X_train, y_train)
```

为什么错：你的 scaler/imputer/encoder 等统计量用到了 test 的信息，这会让验证分数“虚高”。

## 5.6 正确写法：Pipeline 固化边界（推荐你以后都这么写）
原则：**只有 `Pipeline.fit(X_train, y_train)` 会发生 fit**，而且只在训练集上。

## 5.7 直接可用：`train.py`（Titanic + Pipeline）
在 `ml-work/ch05-pipeline/` 新建 `train.py`，运行：

```bash
python train.py
```

```python
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main() -> None:
  url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
  df = pd.read_csv(url)

  y = df["survived"].astype(int)
  feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
  X = df[feature_cols]

  num_cols = ["age", "sibsp", "parch", "fare"]
  cat_cols = ["pclass", "sex", "embarked"]

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
  )

  num_pipe = Pipeline(
    steps=[
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler()),
    ]
  )
  cat_pipe = Pipeline(
    steps=[
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
  )

  preprocess = ColumnTransformer(
    transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)]
  )

  clf = Pipeline(
    steps=[
      ("preprocess", preprocess),
      ("model", LogisticRegression(max_iter=2000)),
    ]
  )

  clf.fit(X_train, y_train)
  proba = clf.predict_proba(X_test)[:, 1]

  roc_auc = roc_auc_score(y_test, proba)
  pr_auc = average_precision_score(y_test, proba)

  print("pipeline: ok")
  print("roc_auc:", round(float(roc_auc), 4))
  print("pr_auc:", round(float(pr_auc), 4))

  Path("models").mkdir(exist_ok=True)
  out_path = Path("models") / "pipeline_logreg.joblib"
  joblib.dump(clf, out_path)
  print("saved:", out_path.as_posix())


if __name__ == "__main__":
  main()
```

## 5.8 直接可用：`leakage_demo.py`（对照实验：错误 vs 正确）
这个脚本会做两次训练并输出两组 AUC：
- “错误做法”：在切分前对全量数据做 one-hot/impute/scale
- “正确做法”：Pipeline

运行：

```bash
python leakage_demo.py
```

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocess(num_cols: list[str], cat_cols: list[str]):
  # 让输出变为 dense，方便不同模型一致处理
  try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
  except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

  num_pipe = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
  )
  cat_pipe = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", ohe)]
  )
  return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def main() -> None:
  url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
  df = pd.read_csv(url)

  y = df["survived"].astype(int)
  feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
  X = df[feature_cols]

  num_cols = ["age", "sibsp", "parch", "fare"]
  cat_cols = ["pclass", "sex", "embarked"]

  # 1) 错误做法：全量 fit_transform 后再 split
  preprocess = make_preprocess(num_cols, cat_cols)
  X_all = preprocess.fit_transform(X)  # ❌ 泄漏风险
  X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
  )
  m1 = LogisticRegression(max_iter=2000)
  m1.fit(X_train, y_train)
  p1 = m1.predict_proba(X_test)[:, 1]
  auc_wrong = roc_auc_score(y_test, p1)

  # 2) 正确做法：Pipeline（只在训练集上 fit）
  preprocess2 = make_preprocess(num_cols, cat_cols)
  X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
  )
  pipe = Pipeline(
    steps=[("preprocess", preprocess2), ("model", LogisticRegression(max_iter=2000))]
  )
  pipe.fit(X_train2, y_train2)
  p2 = pipe.predict_proba(X_test2)[:, 1]
  auc_right = roc_auc_score(y_test2, p2)

  print("auc_wrong_preprocess_first:", round(float(auc_wrong), 4))
  print("auc_right_pipeline:", round(float(auc_right), 4))
  print("note: 如果两者差距很小也正常，关键是你掌握了正确姿势")


if __name__ == "__main__":
  main()
```

## 5.9 验收清单（全部满足）
- 你能解释“泄漏”到底发生在什么位置（一句话即可）
- 你的 `train.py` 用 `Pipeline + ColumnTransformer` 训练，并输出 AUC 指标
- 你的 `README.md` 列出：
  - `num_cols` / `cat_cols`
  - 缺失值策略（median/most_frequent）

完成本章后进入第 6 章（树模型 + 交叉验证 + 误差分析）。

完成本章后进入第 6 章。
