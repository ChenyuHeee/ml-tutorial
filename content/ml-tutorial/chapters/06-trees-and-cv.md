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
