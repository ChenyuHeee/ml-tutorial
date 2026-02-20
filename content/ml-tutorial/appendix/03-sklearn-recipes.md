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

你可以直接参考教程的骨架：
- `ml-tutorial/exercises/sklearn_pipeline_skeleton.py`

## 3) CV 评估模板
- 分类：`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 回归：`KFold(n_splits=5, shuffle=True, random_state=42)`
- `cross_validate(pipe, X, y, scoring=..., n_jobs=-1)`

## 4) 模型保存/加载（必须把预处理一起存）
- 保存：`joblib.dump(pipeline, 'models/model.joblib')`
- 加载：`pipeline = joblib.load(...)`

## 5) 误差分析最小套路
- 分类：
  - 输出 `y_true, y_proba, y_pred`
  - 找 Top 自信但错的样本
- 回归：
  - 输出残差 `y_pred - y_true`
  - 找绝对误差最大 Top-K

> 误差分析是你“变强”的最快路径：它告诉你下一步该补数据、补特征还是换任务定义。
