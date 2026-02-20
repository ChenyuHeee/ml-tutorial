# 第 6 章：树模型与交叉验证（1–2 周）

目标：掌握“能打”的表格数据方法：树模型 + 交叉验证 + 误差分析。

## 6.1 概念最小集
- 决策树：容易过拟合
- 随机森林：bagging 降方差
- GBDT：boosting 降偏差（强力基线）
- 交叉验证：更稳定地估计泛化性能

## 6.2 必做练习
- 在同一数据集上对比：
  1) 逻辑回归（或线性回归）
  2) 随机森林
  3) GBDT（sklearn 的 GradientBoosting 或 HistGradientBoosting）
- 使用 5-fold CV（分类建议 stratified）
- 输出：均值 ± 方差（或标准差）

建议你用 sklearn 的这套组合（更标准也更少坑）：
- 分类：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 回归：`KFold(n_splits=5, shuffle=True, random_state=42)`
- 评估：`cross_validate(...)` 或 `cross_val_score(...)`

输出格式建议（写到终端 + README）：
- `model_name, mean_score, std_score`

## 6.3 误差分析（必须做）
- 找出 Top 错误样本（预测最自信但错了）
- 试着解释：是数据质量问题？特征不足？标签噪声？

你可以用这些问题引导自己写“像研究一样”的误差分析：
- 错得最离谱的样本，是否有缺失/异常？
- 错误集中在哪些人群/区间？（例如年龄段、票价区间、某个类别）
- 这些错误更像是“不可预测噪声”还是“特征缺失导致”？

## 6.4 验收任务
在 `ml-work/ch06-trees-cv/` 里完成：
- `train_cv.py`：输出 3 个模型的 CV 结果
- `error_analysis.md`：写 5 条你观察到的错误模式

完成本章后进入第 7 章。

---

下面是“可直接交付”的版本：你复制代码块、运行，就能完成本章。

## 6.5 直接可用：`train_cv.py`（3 模型 + 5-fold CV）
在 `ml-work/ch06-trees-cv/` 新建 `train_cv.py`，运行：

```bash
python train_cv.py
```

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocess(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
  try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
  except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

  num_pipe = Pipeline(
    steps=[
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler()),
    ]
  )
  cat_pipe = Pipeline(
    steps=[
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("onehot", ohe),
    ]
  )
  return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def summarize(name: str, scores: dict) -> None:
  def fmt(key: str) -> str:
    vals = scores[f"test_{key}"]
    return f"{vals.mean():.4f} ± {vals.std():.4f}"

  print(f"\n[{name}]")
  print("auc:", fmt("auc"))
  print("f1 :", fmt("f1"))


def main() -> None:
  url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
  df = pd.read_csv(url)

  y = df["survived"].astype(int)
  feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
  X = df[feature_cols]

  num_cols = ["age", "sibsp", "parch", "fare"]
  cat_cols = ["pclass", "sex", "embarked"]

  preprocess = make_preprocess(num_cols, cat_cols)
  cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

  models: list[tuple[str, object]] = [
    ("logreg", LogisticRegression(max_iter=2000)),
    ("random_forest", RandomForestClassifier(n_estimators=400, random_state=42)),
    (
      "hist_gbdt",
      HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=6,
        max_iter=300,
        random_state=42,
      ),
    ),
  ]

  scoring = {"auc": "roc_auc", "f1": "f1"}
  for name, model in models:
    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    summarize(name, scores)

  print("\nnote: 只要你能稳定输出 mean±std，并且能解释差异，本章就达标")


if __name__ == "__main__":
  main()
```

你会看到类似输出：

```text
[logreg]
auc: 0.7xxx ± 0.0xxx
f1 : 0.6xxx ± 0.0xxx

[random_forest]
...
```

## 6.6 直接可用：`error_analysis.py`（最自信但错 Top-K）
这一段会让你真正“像做研究一样”进步。

运行：

```bash
python error_analysis.py --model hist_gbdt --k 10
```

```python
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocess(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
  try:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
  except TypeError:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

  num_pipe = Pipeline(
    steps=[
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler()),
    ]
  )
  cat_pipe = Pipeline(
    steps=[
      ("imputer", SimpleImputer(strategy="most_frequent")),
      ("onehot", ohe),
    ]
  )
  return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


@dataclass
class Config:
  model: str = "hist_gbdt"  # logreg | random_forest | hist_gbdt
  k: int = 10


def parse_args() -> Config:
  p = argparse.ArgumentParser()
  p.add_argument("--model", type=str, default="hist_gbdt")
  p.add_argument("--k", type=int, default=10)
  ns = p.parse_args()
  return Config(model=ns.model, k=ns.k)


def make_model(name: str):
  if name == "logreg":
    return LogisticRegression(max_iter=2000)
  if name == "random_forest":
    return RandomForestClassifier(n_estimators=400, random_state=42)
  if name == "hist_gbdt":
    return HistGradientBoostingClassifier(
      learning_rate=0.08, max_depth=6, max_iter=300, random_state=42
    )
  raise ValueError(f"unknown model: {name}")


def main() -> None:
  cfg = parse_args()
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

  preprocess = make_preprocess(num_cols, cat_cols)
  model = make_model(cfg.model)
  pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

  pipe.fit(X_train, y_train)
  proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

  # HistGradientBoosting 没有 predict_proba 时可用 decision_function，但这里它有。
  assert proba is not None

  auc = roc_auc_score(y_test, proba)
  print("model:", cfg.model)
  print("holdout_auc:", round(float(auc), 4))

  pred = (proba >= 0.5).astype(int)
  wrong = np.where(pred != y_test.to_numpy())[0]
  if len(wrong) == 0:
    print("no wrong samples on this split (unlikely).")
    return

  # 最自信但错：距离 0.5 越远越“自信”
  conf = np.abs(proba[wrong] - 0.5)
  top = wrong[np.argsort(conf)[::-1][: cfg.k]]

  out = X_test.iloc[top].copy()
  out["y_true"] = y_test.iloc[top].to_numpy()
  out["y_proba"] = np.round(proba[top], 4)
  out["y_pred@0.5"] = pred[top]

  print("\nTop confident wrong samples:")
  with pd.option_context("display.max_rows", None, "display.width", 140):
    print(out)


if __name__ == "__main__":
  main()
```

## 6.7 验收任务（必须完成）
- `train_cv.py` 能输出 3 个模型的 `mean ± std`（至少 AUC）
- 你写一份 `error_analysis.md`，至少 5 条观察（可以照下面模板写）：

```markdown
# 错误分析（至少 5 条）

1. 最自信但错的样本有什么共同点？
2. 错误是否集中在某些人群（例如年龄段、舱位）？
3. 哪些字段的缺失/异常更容易导致错误？
4. 你猜“不可预测噪声”占比多少？
5. 下一步你会怎么改（补数据/换特征/换切分/换指标）？
```

完成本章后进入第 7 章（无监督：分群与可视化）。
