import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
)
from sklearn.linear_model import LinearRegression, LogisticRegression


@dataclass(frozen=True)
class TrainConfig:
    data_path: str
    target: str
    task: str
    test_size: float
    random_state: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to a CSV file")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument(
        "--task",
        choices=["regression", "binary"],
        required=True,
        help="Task type",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    return TrainConfig(
        data_path=args.data,
        target=args.target,
        task=args.task,
        test_size=args.test_size,
        random_state=args.seed,
    )


def main() -> None:
    cfg = parse_args()

    df = pd.read_csv(cfg.data_path)
    if cfg.target not in df.columns:
        raise ValueError(f"target column not found: {cfg.target}")

    y = df[cfg.target]
    X = df.drop(columns=[cfg.target])

    # 最小版本：只保留数值列（先跑通；后续章节会加入 OneHot 等）
    X = X.select_dtypes(include=[np.number]).copy()

    if X.shape[1] == 0:
        raise ValueError("No numeric features found. Use a different dataset or add preprocessing.")

    stratify = y if cfg.task == "binary" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify,
    )

    if cfg.task == "regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"MAE={mae:.4f} RMSE={rmse:.4f}")
        return

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    print(f"ROC_AUC={roc_auc:.4f} PR_AUC={pr_auc:.4f}")


if __name__ == "__main__":
    main()
