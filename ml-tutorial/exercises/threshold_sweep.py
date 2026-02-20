import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="CSV with columns: y_true, y_proba")
    parser.add_argument(
        "--thresholds",
        default="0.1,0.3,0.5,0.7,0.9",
        help="comma-separated thresholds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.labels)

    if not {"y_true", "y_proba"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: y_true, y_proba")

    y_true = df["y_true"].to_numpy()
    y_proba = df["y_proba"].to_numpy()

    thresholds = [float(x) for x in args.thresholds.split(",")]

    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
        print(f"threshold={thr:.2f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
        print("confusion_matrix=")
        print(cm)


if __name__ == "__main__":
    main()
