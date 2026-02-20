# 第 5 章：特征工程与数据泄漏（sklearn Pipeline）（1–2 周）

目标：你能用 Pipeline 把预处理与模型训练“绑定在一起”，彻底避免常见泄漏。

## 5.1 概念最小集
- 数据泄漏（data leakage）：训练时看到了不该看的信息
- 正确做法：所有从数据学出来的变换（填充、缩放、编码）必须只在训练集上 fit
- Pipeline 的价值：把 fit/transform 的边界固化

## 5.2 你要掌握的 sklearn 组件
- `train_test_split`
- `ColumnTransformer`
- `SimpleImputer`（缺失值）
- `OneHotEncoder`（类别编码）
- `StandardScaler`（数值缩放）
- `Pipeline`

你最终要长成的结构（先看懂，再照着写）：
- `preprocess = ColumnTransformer([...])`
- `model = LogisticRegression(...)`（或回归模型）
- `clf = Pipeline([('preprocess', preprocess), ('model', model)])`

这样做的好处：任何 `fit` 都会先在训练集上 fit 预处理器；你不会“不小心”用全量数据统计均值/方差。

## 5.3 必做练习
- 用你第 4 章的数据，把预处理全部迁移到 Pipeline
- 对比：
  - 不用 Pipeline（容易泄漏）
  - 用 Pipeline（正确）
- 你要输出：同样的验证方式下，两者指标是否变化？为什么？

你要在 README 里回答这两个问题：
1. 你之前的写法哪里可能发生泄漏？（哪一步在全量数据上学了统计量？）
2. Pipeline 写法如何强制你在训练集上 fit？

## 5.4 验收任务
在 `ml-work/ch05-pipeline/` 里完成：
- `train.py`：使用 Pipeline 训练并输出指标
- `README.md`：列出数值/类别特征列表，并解释你的缺失值策略

完成本章后进入第 6 章。
