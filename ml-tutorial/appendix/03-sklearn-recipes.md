# 附录 03：sklearn 实战配方（能直接抄）

## 1) 标准项目结构（建议）
- `src/train.py`：训练 + 保存模型
- `src/evaluate.py`：评估 + 指标输出
- `src/predict.py`：推理（读 CSV → 输出预测）
- `models/`：保存模型文件
- `reports/`：图表、误差分析

## 2) Pipeline 模板（数值 + 类别）
核心原则：**所有会“学习统计量”的步骤必须在训练集 fit**。

- `SimpleImputer(median/most_frequent)`
- `StandardScaler()`（数值）
- `OneHotEncoder(handle_unknown='ignore')`（类别）
- `Pipeline([('preprocess', ...), ('model', ...)])`

### 可直接复制的最小 Pipeline（数值+类别）
把下面代码保存为 `train.py`（文件名你也可以自己改），然后按你的数据列名替换 `num_cols/cat_cols/label_col`。

```python
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main() -> None:
  data_path = Path("clean.csv")  # 改成你的数据路径
  df = pd.read_csv(data_path)

  label_col = "survived"  # 改成你的标签列
  y = df[label_col].astype(int)
  X = df.drop(columns=[label_col])

  num_cols = ["age", "fare", "sibsp", "parch"]  # 改成你的数值列
  cat_cols = ["sex", "embarked", "pclass"]       # 改成你的类别列

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
    transformers=[
      ("num", num_pipe, num_cols),
      ("cat", cat_pipe, cat_cols),
    ],
    remainder="drop",
  )

  model = LogisticRegression(max_iter=1000)

  pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
  pipe.fit(X_train, y_train)

  proba = pipe.predict_proba(X_test)[:, 1]
  auc = roc_auc_score(y_test, proba)
  print("roc_auc:", round(auc, 4))

  Path("models").mkdir(exist_ok=True)
  joblib.dump(pipe, "models/model.joblib")
  print("saved: models/model.joblib")


if __name__ == "__main__":
  main()
```

## 3) CV 评估模板
- 分类：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 回归：`KFold(n_splits=5, shuffle=True, random_state=42)`
- `cross_validate(pipe, X, y, scoring=..., n_jobs=-1)`

最小示例（把 `pipe/X/y` 换成你的）：

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(pipe, X, y, cv=cv, scoring={"auc": "roc_auc"}, n_jobs=-1)
print("cv_auc_mean:", scores["test_auc"].mean())
print("cv_auc_std:", scores["test_auc"].std())
```

## 4) 模型保存/加载（必须把预处理一起存）
- 保存：`joblib.dump(pipeline, 'models/model.joblib')`
- 加载：`pipeline = joblib.load(...)`

加载后直接预测（保持和训练一致的预处理）：

```python
import joblib
import pandas as pd

pipe = joblib.load("models/model.joblib")
df = pd.read_csv("new_data.csv")
proba = pipe.predict_proba(df)[:, 1]
print(proba[:10])
```

## 5) 误差分析最小套路
- 分类：
  - 输出 `y_true, y_proba, y_pred`
  - 找 Top 自信但错的样本
- 回归：
  - 输出残差 `y_pred - y_true`
  - 找绝对误差最大 Top-K

> 误差分析是你“变强”的最快路径：它告诉你下一步该补数据、补特征还是换任务定义。

分类的最小误差分析（找“最自信但错”的样本）：

```python
import numpy as np

proba = pipe.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)
wrong = np.where(pred != y_test.to_numpy())[0]

# 只看 Top-K（最自信但错）
k = 10
idx = wrong[np.argsort(np.abs(proba[wrong] - 0.5))[::-1][:k]]
print("top_confident_wrong_idx:", idx.tolist())
print(X_test.iloc[idx])
print("y_true:", y_test.iloc[idx].tolist())
print("y_proba:", proba[idx].round(4).tolist())
```
