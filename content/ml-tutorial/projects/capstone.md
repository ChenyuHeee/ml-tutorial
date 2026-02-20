# 最终项目（Capstone）：做一个作品集级端到端项目（2–4 周）

目标：把你学到的“数据→训练→评估→误差分析→推理”串成一个可展示的项目。

## 选题要求（尽量满足）
- 数据量：>= 1k 样本（太小很难泛化）
- 任务：二分类/多分类/回归 任意
- 评价：指标可量化（例如 AUC、F1、MAE）
- 复现：别人能按 README 跑通

## 推荐选题（你也可以自选）
- 表格数据：信用/风控、用户流失、房价/销量预测
- 文本：评论情感、工单分类、垃圾邮件
- 图像：小型分类（自建数据也可以）

## 项目目录建议
- `ml-work/capstone/`
  - `data/`（可选：放下载脚本，不一定放原始大文件）
  - `src/`
    - `train.py`
    - `evaluate.py`
    - `predict.py`
  - `models/`（保存模型）
  - `reports/`（图表、误差分析）
  - `README.md`

## 里程碑（按顺序完成）
### M1：问题与数据（0.5–1 天）
- 写清楚：预测目标、输入特征、潜在泄漏
- 做一个基线：多数类/均值预测

### M2：经典 ML baseline（2–4 天）
- 用 Pipeline 完成预处理 + 模型（逻辑回归/随机森林/GBDT）
- 用 CV 或稳定 holdout 输出指标
- 误差分析：列出 Top 错误样本/切片

### M3：改进（3–7 天）
从下面选 2–3 个做“可验证”的改进：
- 更合理的特征工程（尤其是类别特征）
- 更合理的验证方式（分层/时间切分）
- 更合理的阈值（分类）
- 调参（随机搜索）
- 引入更强模型（GBDT 或简单神经网络）

### M4：推理脚本与复现（1–2 天）
- `predict.py`：给定一条样本或一个 CSV 输出预测
- README 写清楚：安装、训练、评估、推理

## 验收清单（全部满足才算完成）
- 能一键训练并输出指标
- 能跑推理脚本并得到结果
- README 解释清楚：指标、验证方式、误差分析、下一步

你完成 M1 后，把你的“问题定义 + 数据链接 + 指标与切分方案”发我，我会按你的题目继续带你推进。

---

下面是“作品集级”的最小脚手架：你把自己的数据替换进去，就能形成一个别人也能复现的端到端项目。

## A) 快速开始（没有自己的数据也能先跑通）
如果你暂时没选好数据集，可以先用 California Housing（回归，`sklearn` 自带/自动下载）跑通整个 capstone 目录。

在 `ml-work/capstone/` 下执行：

```bash
mkdir -p data src models reports
python - <<'PY'
from pathlib import Path
import pandas as pd
from sklearn.datasets import fetch_california_housing

ds = fetch_california_housing(as_frame=True)
df = ds.frame
df.to_csv(Path('data')/'train.csv', index=False)
print('wrote:', Path('data')/'train.csv', 'shape=', df.shape)
PY


说明：这会生成 `data/train.csv`，目标列是 `MedHouseVal`。

## B) 目录结构（就按这个交付）

```text
ml-work/capstone/
  data/
  train.csv
  test.csv            # 可选：你也可以只有 train.csv
  src/
  train.py
  evaluate.py
  predict.py
  models/
  model.joblib
  reports/
  metrics.json
  top_errors.csv
  run_config.json
  README.md
```

## C) 直接可用：`src/train.py`（训练 + holdout 评估 + 保存模型 + 误差分析）
把下面代码保存为 `src/train.py`：

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
  accuracy_score,
  f1_score,
  mean_absolute_error,
  mean_squared_error,
  r2_score,
  roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
  cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
  num_cols = [c for c in X.columns if c not in cat_cols]

  numeric = Pipeline(
    steps=[
      ("imputer", SimpleImputer(strategy="median")),
      ("scaler", StandardScaler()),
    ]
  )
  categorical = Pipeline(
    steps=[
      ("imputer", SimpleImputer(strategy="most_frequent")),
      (
        "onehot",
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
      ),
    ]
  )

  return ColumnTransformer(
    transformers=[
      ("num", numeric, num_cols),
      ("cat", categorical, cat_cols),
    ],
    remainder="drop",
  )


def infer_problem(y: pd.Series, forced: str | None) -> str:
  if forced is not None:
    return forced
  # 很粗糙但够用：数值且 unique 很多 -> 回归；否则分类
  if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 20:
    return "regression"
  return "classification"


def classification_metrics(y_true: np.ndarray, proba: np.ndarray | None, y_pred: np.ndarray) -> dict[str, Any]:
  out: dict[str, Any] = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
  }
  # 二分类且有 proba 才算 AUC
  if proba is not None and len(np.unique(y_true)) == 2:
    out["roc_auc"] = float(roc_auc_score(y_true, proba))
  return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
  rmse = mean_squared_error(y_true, y_pred, squared=False)
  return {
    "mae": float(mean_absolute_error(y_true, y_pred)),
    "rmse": float(rmse),
    "r2": float(r2_score(y_true, y_pred)),
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--train_csv", default="data/train.csv")
  ap.add_argument("--target", required=True)
  ap.add_argument("--problem", choices=["classification", "regression"], default=None)
  ap.add_argument("--test_size", type=float, default=0.2)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--model_out", default="models/model.joblib")
  ap.add_argument("--reports_dir", default="reports")
  args = ap.parse_args()

  train_path = Path(args.train_csv)
  df = pd.read_csv(train_path)
  if args.target not in df.columns:
    raise SystemExit(f"target column not found: {args.target}")

  y = df[args.target]
  X = df.drop(columns=[args.target])

  problem = infer_problem(y, args.problem)
  print("problem:", problem)

  stratify = y if problem == "classification" else None
  X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=args.test_size,
    random_state=args.seed,
    stratify=stratify,
  )

  preprocess = build_preprocess(X_train)
  if problem == "classification":
    model = LogisticRegression(max_iter=2000)
  else:
    model = Ridge(alpha=1.0, random_state=args.seed)

  pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
  pipe.fit(X_train, y_train)

  # 评估
  if problem == "classification":
    y_pred = pipe.predict(X_val)
    proba = None
    if hasattr(pipe, "predict_proba"):
      proba = pipe.predict_proba(X_val)[:, 1] if len(np.unique(y_val)) == 2 else None
    metrics = classification_metrics(np.asarray(y_val), proba, np.asarray(y_pred))

    # 误差分析：最自信但错的样本
    val_out = X_val.copy()
    val_out["y_true"] = y_val.values
    val_out["y_pred"] = y_pred
    if proba is not None:
      val_out["p_pos"] = proba
      wrong = val_out[val_out["y_true"] != val_out["y_pred"]].copy()
      wrong["confidence"] = np.maximum(wrong["p_pos"], 1.0 - wrong["p_pos"])
      top_errors = wrong.sort_values("confidence", ascending=False).head(30)
    else:
      top_errors = val_out[val_out["y_true"] != val_out["y_pred"]].head(30)
  else:
    y_pred = pipe.predict(X_val)
    metrics = regression_metrics(np.asarray(y_val), np.asarray(y_pred))

    # 误差分析：绝对误差最大的样本
    val_out = X_val.copy()
    val_out["y_true"] = y_val.values
    val_out["y_pred"] = y_pred
    val_out["abs_err"] = np.abs(val_out["y_true"] - val_out["y_pred"])
    top_errors = val_out.sort_values("abs_err", ascending=False).head(30)

  print("metrics:", metrics)

  # 落盘
  model_out = Path(args.model_out)
  model_out.parent.mkdir(parents=True, exist_ok=True)
  dump(pipe, model_out)

  reports_dir = Path(args.reports_dir)
  reports_dir.mkdir(parents=True, exist_ok=True)

  (reports_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
  top_errors.to_csv(reports_dir / "top_errors.csv", index=False)

  run_cfg = {
    "train_csv": str(train_path),
    "target": args.target,
    "problem": problem,
    "test_size": args.test_size,
    "seed": args.seed,
    "model_out": str(model_out),
  }
  (reports_dir / "run_config.json").write_text(json.dumps(run_cfg, ensure_ascii=False, indent=2))

  print("saved model:", model_out)
  print("saved reports:", reports_dir)


if __name__ == "__main__":
  main()
```

运行方式（quickstart 用 California Housing）：

```bash
python src/train.py --target MedHouseVal --problem regression
```

如果你是二分类数据：

```bash
python src/train.py --train_csv data/train.csv --target target --problem classification
```

## D) 直接可用：`src/evaluate.py`（对外部 test.csv 做最终评估）
把下面保存为 `src/evaluate.py`：

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import (
  accuracy_score,
  f1_score,
  mean_absolute_error,
  mean_squared_error,
  r2_score,
  roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, proba: np.ndarray | None, y_pred: np.ndarray) -> dict[str, Any]:
  out: dict[str, Any] = {
    "accuracy": float(accuracy_score(y_true, y_pred)),
    "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
  }
  if proba is not None and len(np.unique(y_true)) == 2:
    out["roc_auc"] = float(roc_auc_score(y_true, proba))
  return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
  rmse = mean_squared_error(y_true, y_pred, squared=False)
  return {
    "mae": float(mean_absolute_error(y_true, y_pred)),
    "rmse": float(rmse),
    "r2": float(r2_score(y_true, y_pred)),
  }


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--test_csv", default="data/test.csv")
  ap.add_argument("--target", required=True)
  ap.add_argument("--problem", choices=["classification", "regression"], required=True)
  ap.add_argument("--model", default="models/model.joblib")
  ap.add_argument("--out", default="reports/test_metrics.json")
  args = ap.parse_args()

  df = pd.read_csv(args.test_csv)
  if args.target not in df.columns:
    raise SystemExit(f"target column not found: {args.target}")
  y = df[args.target]
  X = df.drop(columns=[args.target])

  pipe = load(args.model)
  if args.problem == "classification":
    y_pred = pipe.predict(X)
    proba = None
    if hasattr(pipe, "predict_proba") and len(np.unique(y)) == 2:
      proba = pipe.predict_proba(X)[:, 1]
    metrics = classification_metrics(np.asarray(y), proba, np.asarray(y_pred))
  else:
    y_pred = pipe.predict(X)
    metrics = regression_metrics(np.asarray(y), np.asarray(y_pred))

  out_path = Path(args.out)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
  print("test metrics:", metrics)
  print("saved:", out_path)


if __name__ == "__main__":
  main()
```

## E) 直接可用：`src/predict.py`（对无标签数据做推理）
把下面保存为 `src/predict.py`：

```python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from joblib import load


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--input_csv", required=True)
  ap.add_argument("--output_csv", default="predictions.csv")
  ap.add_argument("--model", default="models/model.joblib")
  ap.add_argument("--drop", default=None, help="可选：要丢掉的列名（比如 id 列），逗号分隔")
  args = ap.parse_args()

  df = pd.read_csv(args.input_csv)
  if args.drop:
    drop_cols = [c.strip() for c in args.drop.split(",") if c.strip()]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

  pipe = load(args.model)
  pred = pipe.predict(df)

  out = df.copy()
  out["prediction"] = pred
  if hasattr(pipe, "predict_proba"):
    try:
      proba = pipe.predict_proba(df)
      if proba.shape[1] == 2:
        out["p_pos"] = proba[:, 1]
    except Exception:
      pass

  out_path = Path(args.output_csv)
  out.to_csv(out_path, index=False)
  print("saved:", out_path)


if __name__ == "__main__":
  main()
```

## F) README 模板（别人照着就能复现）
把下面保存为 `README.md`（按你的项目改名即可）：

````markdown
# Capstone：<你的项目名>

## 1) 任务定义
- 输入：
- 输出（target）：
- 业务意义：

## 2) 数据
- 数据来源：
- 样本量：
- 泄漏风险：

## 3) 指标与切分
- 指标：
- 切分方式：随机 / 分层 / 时间切分（理由）：

## 4) 复现方式

### 安装
（写你的环境方式；如果是本教程仓库，说明 requirements/conda 环境即可）

### 训练
```bash
python src/train.py --train_csv data/train.csv --target <target> --problem <classification|regression>
```

### 评估（可选：有 test.csv 才跑）
```bash
python src/evaluate.py --test_csv data/test.csv --target <target> --problem <classification|regression>
```

### 推理
```bash
python src/predict.py --input_csv data/unlabeled.csv --output_csv predictions.csv
```

## 5) 结果与误差分析
- 指标（见 reports/metrics.json）：
- Top 错误样本（见 reports/top_errors.csv）：

## 6) 下一步改进
- （列 3 条）
````

## G) 里程碑验收（照着打勾就能收敛）
- [ ] M1：写清楚 target/特征/泄漏风险；跑通基线（多数类/均值）
- [ ] M2：跑通 `src/train.py`，产出 `models/model.joblib` + `reports/metrics.json`
- [ ] M2：写出 5 条误差分析结论（来自 `reports/top_errors.csv`）
- [ ] M3：做 2 个可验证改进（特征/切分/阈值/调参/更强模型），对比指标
- [ ] M4：推理脚本可用 + README 可复现（陌生人照着能跑通）
