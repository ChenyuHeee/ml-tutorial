# 第 4 章：逻辑回归与二分类评估（1 周）

目标：你能完成一个二分类任务，正确使用概率输出、阈值与分类指标。

## 4.1 概念最小集
- 逻辑回归输出的是概率 $p(y=1|x)$
- 损失：对数损失（交叉熵）
- 阈值：把概率变成类别（默认 0.5 不是永远最佳）

## 4.2 指标与图
- 混淆矩阵
- Precision/Recall/F1
- ROC 曲线、PR 曲线

## 4.3 必做练习
- 选一个二分类数据集：Titanic 是经典
- 用逻辑回归做 baseline
- 输出：
  - ROC-AUC 与 PR-AUC
  - 在不同阈值下的 Precision/Recall/F1（至少比较 3 个阈值）

建议你按这个顺序做（不容易乱）：
1. `train_test_split(..., stratify=y, random_state=42)`（分类尽量分层）
2. 先只训练一个 baseline（逻辑回归）
3. 用 `predict_proba` 得到概率，再做：
  - AUC（不需要阈值）
  - 阈值扫描（需要你自己遍历阈值）

## 4.4 验收任务
在 `ml-work/ch04-logistic-regression/` 里完成：
- `train.py`：训练并输出 ROC-AUC/PR-AUC/F1
- `evaluate.py`：给定阈值，输出混淆矩阵与指标
- `README.md`：说明你选阈值的依据（例如更重视 Recall）

验收标准：
- 你能说清楚：你在什么业务假设下更重视 Precision 或 Recall
- 你的 `evaluate.py --threshold 0.3`（或类似方式）能输出混淆矩阵

## 4.5 常见坑
- 只看 accuracy（类别不均衡时会自欺欺人）
- 没有分层抽样导致验证集类别比例异常

额外提醒：
- 用 AUC 报告模型“排序能力”，用阈值后的 F1/Recall 报告“决策效果”。两者不是一回事。

完成本章后进入第 5 章。
