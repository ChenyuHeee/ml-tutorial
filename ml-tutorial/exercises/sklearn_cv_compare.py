import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


@dataclass(frozen=True)
class Config:
    data: str
    target: str
    task: str
    seed: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", choices=["binary", "regression"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return Config(data=args.data, target=args.target, task=args.task, seed=args.seed)


def main() -> None:
    cfg = parse_args()

    df = pd.read_csv(cfg.data)
    y = df[cfg.target]
    X = df.drop(columns=[cfg.target])

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    if cfg.task == "binary":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed)
        scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
        models = {
            "logreg": LogisticRegression(max_iter=400),
            "rf": RandomForestClassifier(n_estimators=400, random_state=cfg.seed),
            "hgb": HistGradientBoostingClassifier(random_state=cfg.seed),
        }
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=cfg.seed)
        scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error"}
        models = {"ridge": Ridge(alpha=1.0, random_state=cfg.seed)}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        print(f"\n[{name}]")
        for metric, values in scores.items():
            if not metric.startswith("test_"):
                continue
            m = metric.replace("test_", "")
            v = values
            # regression metrics are negative in sklearn (when using neg_* scorers)
            if cfg.task == "regression":
                v = -v
            print(f"{m}: mean={np.mean(v):.4f} std={np.std(v):.4f}")


if __name__ == "__main__":
    main()
