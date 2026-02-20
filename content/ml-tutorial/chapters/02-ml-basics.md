# 第 2 章：机器学习问题是什么（1 周）

目标：你能把一个现实问题写成机器学习任务，并正确做数据划分与指标选择。

本章你会第一次把“现实问题 → 数据表 → 模型输入输出 → 评估方式”写清楚。这个能力比会调模型更重要。

## 2.1 概念最小集
- 样本（sample）、特征（features）、标签（label）
- 监督学习：回归/分类；无监督：聚类/降维
- 训练集/验证集/测试集：
  - 测试集只用一次：最终报告
- 泛化：在新数据上表现好

## 2.2 指标（先记住这些）
### 分类
- Accuracy：只在类别均衡时有意义
- Precision/Recall/F1：类别不均衡时更可靠
- ROC-AUC vs PR-AUC：极度不均衡时 PR-AUC 更敏感

补一句“怎么用”（先记住就够）：
- 你在做**概率输出**（`predict_proba`）时，优先看 `ROC-AUC/PR-AUC`（阈值无关）
- 你要落地一个“判定规则”（阈值）时，再看 `Precision/Recall/F1`

指标与验证的速查表（很实用）：[../appendix/01-metrics-and-validation.md](../appendix/01-metrics-and-validation.md)

### 回归
- MAE：对异常值更稳
- RMSE：更惩罚大误差

## 2.3 验证方式（重要）
- 普通任务：K 折交叉验证（或 train/val/test）
- 时间序列：按时间切分（不能随机打乱）
- 分层抽样：分类任务保持类别比例

你先记住一句话：**切分方式要模拟真实上线场景**（否则验证分数没有意义）。

## 2.4 必做练习
用你第 1 章的数据（或另选一个）：
1. 明确任务：你要预测什么？为什么有价值？
2. 指定输入/输出：哪些列能用、哪些列不能用（潜在泄漏）
3. 设计评估：选择指标 + 切分方案
4. 写一个“基线方案”：
   - 例如：回归用预测均值；分类用预测多数类

建议你本章练习目录为：`ml-work/ch02-ml-basics/`，并至少产出：
- `README.md`：任务定义（按模板填满）
- `baseline.py`：可运行的基线（输出指标）

### 一个完整示例（Titanic，推荐你先照抄再改）
如果你第 1 章做的是 Titanic，你可以把任务定义写成这样：

- 任务类型：二分类
- 标签定义：`survived`（1=幸存，0=罹难）
- 输入特征（允许）：`pclass, sex, age, sibsp, parch, fare, embarked`
- 你希望优化的指标：先用 `ROC-AUC` 衡量排序能力；落阈值后看 `F1`
- 数据切分方式：随机分层切分（`stratify=y`），并固定 `random_state=42`

为什么这个示例好：它足够小，能让你把流程跑通；后面第 3–6 章所有内容都能在它上面加。

建议你直接按模板写（复制到你的 README）：

### 任务描述模板
- 任务类型：二分类 / 多分类 / 回归
- 标签定义：
- 你希望优化的指标：
- 为什么这个指标合理：
- 数据切分方式：随机分层 / 时间切分 / Group 切分（如按用户）

### 泄漏自检清单（逐条过一遍）
- 是否存在“事后才知道”的字段（例如是否退款、是否死亡、最终成绩等）？
- 是否在切分之前对全量数据做了统计/编码/缩放？
- 是否把同一个人的多条记录同时放进 train 和 test？（需要 Group 切分）

更系统的泄漏排查：见附录 → [../appendix/02-leakage-checklist.md](../appendix/02-leakage-checklist.md)

## 2.5 验收任务
- 在 `ml-work/ch02-ml-basics/README.md` 写清楚：
  - 任务类型、标签定义
  - 你选择的指标与理由
  - 你采用的切分方式与理由
  - 基线方法的指标（哪怕很差）

基线怎么做（你照着实现即可）：
- 回归：`y_pred = y_train.mean()`
- 分类：永远预测训练集多数类

### `baseline.py` 最小可运行模板（分类：Titanic）
把下面代码保存为 `ml-work/ch02-ml-basics/baseline.py`（路径你可自行调整），然后在终端运行：`python baseline.py`。

```python
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main() -> None:
  # 建议用你自己在第 1 章产出的 clean.csv
  data_path = Path(__file__).resolve().parent / "../ch01-python-data/clean.csv"
  df = pd.read_csv(data_path)

  y = df["survived"].astype(int)
  X = df.drop(columns=["survived"])

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
  )

  majority = int(y_train.value_counts().idxmax())
  y_pred = [majority] * len(y_test)
  acc = accuracy_score(y_test, y_pred)

  print("majority_class:", majority)
  print("accuracy:", round(acc, 4))


if __name__ == "__main__":
  main()
```

注意：这个基线只输出 Accuracy（因为它只给 hard label）。到第 4 章你会换成 `predict_proba` 并学习 AUC/阈值。

完成本章后进入第 3 章。
