# 第 7 章：无监督（聚类与降维可视化）（1 周）

目标：你能做一个“分群/聚类”分析，并输出可解释的簇画像。

## 7.1 概念最小集
- 聚类不是分类：没有标签，评估更依赖解释与稳定性
- KMeans：对尺度敏感，需要标准化
- PCA：用于降维与解释方差

## 7.2 必做练习
- 选一个数据集（可以用你前面的 clean 数据）：
  - 先做标准化
  - 用 PCA 降到 2 维做可视化
  - 用 KMeans 做聚类（尝试 k=2..8）
- 输出：
  - 不同 k 的结果对比（你可以用轮廓系数做参考，但不要迷信）
  - 每个簇的“均值特征画像”

## 7.3 验收任务
在 `ml-work/ch07-unsupervised/` 里完成：
- 一页报告（markdown）：
  - 你选 k 的理由
  - 每个簇的画像
  - 你认为这个分群能解决什么问题

完成本章后进入第 8 章。

---

本章用一个无需下载文件的数据集帮你把“无监督完整流程”跑通：Wine 数据集（`sklearn.datasets.load_wine`）。

## 7.4 直接可用：`cluster.py`（标准化 → PCA2 可视化 → KMeans → 画像）
在 `ml-work/ch07-unsupervised/` 新建 `cluster.py`，运行：

```bash
python cluster.py


```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main() -> None:
  ds = load_wine(as_frame=True)
  X = ds.data

  scaler = StandardScaler()
  Xs = scaler.fit_transform(X)

  # 1) PCA 到 2D 方便可视化
  pca = PCA(n_components=2, random_state=42)
  Z = pca.fit_transform(Xs)
  print("pca_explained_variance_ratio:", np.round(pca.explained_variance_ratio_, 4).tolist())

  # 2) 尝试不同 k
  rows = []
  best = (None, -1.0)
  for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    sil = silhouette_score(Xs, labels)
    rows.append({"k": k, "silhouette": float(sil)})
    if sil > best[1]:
      best = (k, float(sil))

  table = pd.DataFrame(rows).sort_values("k")
  print("\nSilhouette by k:\n", table)
  k_best = int(best[0])
  print("\nbest_k:", k_best, "best_silhouette:", round(best[1], 4))

  # 3) 用 best_k 聚类 + 画像
  km = KMeans(n_clusters=k_best, n_init=10, random_state=42)
  labels = km.fit_predict(Xs)

  df = X.copy()
  df["cluster"] = labels

  profile = df.groupby("cluster").mean(numeric_only=True)
  print("\nCluster profile (mean):\n", profile)

  Path("reports").mkdir(exist_ok=True)

  # 4) 可视化（PCA2 + 聚类颜色）
  plt.figure(figsize=(6, 5))
  for c in sorted(np.unique(labels)):
    idx = labels == c
    plt.scatter(Z[idx, 0], Z[idx, 1], s=18, label=f"cluster {c}")
  plt.title(f"Wine clustering (k={k_best})")
  plt.xlabel("PC1")
  plt.ylabel("PC2")
  plt.legend(frameon=False)
  out_path = Path("reports") / "pca_kmeans.png"
  plt.tight_layout()
  plt.savefig(out_path, dpi=160)
  print("\nsaved:", out_path.as_posix())


if __name__ == "__main__":
  main()
```

## 7.5 直接可用：`report.md` 模板（一页就够）
把下面内容保存为 `report.md`，根据你跑出来的结果补全。

```markdown
# 无监督分群报告（Wine）

## 1) 我为什么要做分群
- 目标：把样本分成若干组，让每组内部更相似、组间更不同
- 预期用途：人群画像、运营策略、异常检测的“正常模式”参考

## 2) 我怎么做的
- 标准化：StandardScaler（KMeans 对尺度敏感）
- PCA：降到 2D 用于可视化（不是为了更准，是为了更好解释）
- KMeans：尝试 k=2..8，用 silhouette 做参考

## 3) 我怎么选 k
- 我选的 k：
- 理由：silhouette 在该 k 附近较好 + 可解释性更强（不要只迷信 silhouette）

## 4) 簇画像（每簇 3–5 个特征就够）
- cluster 0：
- cluster 1：
- ...

## 5) 我觉得这个分群能解决什么问题
- （写 3–5 行）
```

## 7.6 验收清单（全部满足）
- `python cluster.py` 能输出 `best_k`、`silhouette` 表，并生成 `reports/pca_kmeans.png`
- `report.md` 里写清楚：选 k 的理由 + 至少 2 个簇的画像

完成本章后进入第 8 章（PyTorch：训练循环与过拟合诊断）。
