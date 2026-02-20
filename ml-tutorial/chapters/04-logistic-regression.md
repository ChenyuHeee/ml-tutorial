# 第 4 章：逻辑回归与二分类评估（1 周）

目标：你能完成一个二分类任务，并且**正确使用概率输出、阈值与分类指标**（而不是只看 accuracy）。

本章使用无需注册的数据：Titanic（公开 CSV）。

```text
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
```

## 4.1 概念最小集（够你把流程跑通）
- 逻辑回归输出的是概率 $p(y=1|x)$（通过 `predict_proba` 拿到）
- 损失：对数损失（交叉熵）
- 阈值：把概率变成类别（默认 0.5 不一定最佳）

你要分清两类指标：
- **阈值无关**：ROC-AUC、PR-AUC（衡量“排序能力”）
- **阈值相关**：Precision/Recall/F1、混淆矩阵（衡量“决策效果”）

## 4.2 本章交付物
在你的练习目录（推荐：`ml-work/ch04-logistic-regression/`）里，至少有：
- `train.py`：训练逻辑回归 + 输出 AUC 指标 + 保存模型
- `evaluate.py`：给定阈值输出混淆矩阵、Precision/Recall/F1
- `threshold_sweep.py`：扫描阈值并输出一个小表（帮助你选阈值）

## 4.3 直接可用：`train.py`（复制即可运行）
运行：

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

  # 标签
  y = df["survived"].astype(int)

  # 一个足够小且经典的特征集合（先跑通，再扩展）
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

  model = LogisticRegression(max_iter=2000)
  clf = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

  clf.fit(X_train, y_train)
  proba = clf.predict_proba(X_test)[:, 1]

  roc_auc = roc_auc_score(y_test, proba)
  pr_auc = average_precision_score(y_test, proba)

  print("dataset: titanic")
  print("n_train:", len(X_train), "n_test:", len(X_test))
  print("roc_auc:", round(float(roc_auc), 4))
  print("pr_auc:", round(float(pr_auc), 4))

  Path("models").mkdir(exist_ok=True)
  out_path = Path("models") / "logreg_titanic.joblib"
  joblib.dump(clf, out_path)
  print("saved:", out_path.as_posix())


if __name__ == "__main__":
  main()
```

预期输出长这样（数值会不同，但结构应一致）：

```text
dataset: titanic
n_train: ... n_test: ...
roc_auc: ...
pr_auc: ...
saved: models/logreg_titanic.joblib
```

## 4.4 直接可用：`evaluate.py`（阈值 → 混淆矩阵与指标）
运行：

```bash
python evaluate.py --threshold 0.3
python evaluate.py --threshold 0.5
```

```python
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser()
  p.add_argument("--threshold", type=float, default=0.5)
  p.add_argument("--model", type=str, default="models/logreg_titanic.joblib")
  return p.parse_args()


def main() -> None:
  args = parse_args()

  url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
  df = pd.read_csv(url)

  y = df["survived"].astype(int)
  feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
  X = df[feature_cols]

  _, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
  )

  clf = joblib.load(Path(args.model))
  proba = clf.predict_proba(X_test)[:, 1]
  pred = (proba >= args.threshold).astype(int)

  print("threshold:", args.threshold)
  print("confusion_matrix:\n", confusion_matrix(y_test, pred))
  print(
    classification_report(
      y_test,
      pred,
      digits=4,
      target_names=["died(0)", "survived(1)"],
    )
  )


if __name__ == "__main__":
  main()
```

## 4.5 直接可用：`threshold_sweep.py`（帮你选阈值）
运行：

```bash
python threshold_sweep.py
```

```python
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def main() -> None:
  url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
  df = pd.read_csv(url)

  y = df["survived"].astype(int)
  feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
  X = df[feature_cols]
  _, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
  )

  clf = joblib.load(Path("models") / "logreg_titanic.joblib")
  proba = clf.predict_proba(X_test)[:, 1]

  thresholds = np.linspace(0.1, 0.9, 9)
  rows = []
  for t in thresholds:
    pred = (proba >= t).astype(int)
    rows.append(
      {
        "threshold": round(float(t), 2),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "pos_rate": float(pred.mean()),
      }
    )

  out = pd.DataFrame(rows)
  with pd.option_context("display.max_rows", None, "display.width", 120):
    print(out)


if __name__ == "__main__":
  main()
```

你会得到一个小表，帮助你在“更高 Recall”与“更高 Precision”之间做取舍。

## 4.6 验收任务（必须完成）
- `python train.py` 输出 `roc_auc/pr_auc` 并保存模型
- `python evaluate.py --threshold 0.3` 能输出混淆矩阵 + Precision/Recall/F1
- `python threshold_sweep.py` 能输出阈值扫描表
- 你在 `README.md` 里写清楚：
  - 你更重视 Precision 还是 Recall（以及你假设的业务理由）
  - 你最终选了哪个阈值

## 4.7 常见坑
- 只看 accuracy（类别不均衡时会自欺欺人）
- 没有 `stratify=y` 导致验证集类别比例异常
- 只汇报 AUC 不汇报阈值后的指标：上线需要“决策效果”

完成本章后进入第 5 章（用 Pipeline 正式解决预处理与泄漏）。
