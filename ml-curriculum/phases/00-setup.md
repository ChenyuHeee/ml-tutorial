# 阶段 0：工具与基础（1 周）

目标：把开发环境与数据分析基本功打牢，能做“可复现”的小型数据分析。

## 必会工具
- Python 环境：`conda` 或 `venv`
- Notebook：Jupyter（用于探索）
- 代码与复现：Git（提交记录）、requirements/环境文件
- 数据处理与可视化：`numpy`、`pandas`、`matplotlib`（可选 `seaborn`）

## 练习清单（按顺序）
1. 用 `pandas` 读入 CSV/Parquet（任意公开数据集）
2. 完成清洗：缺失值处理、类型转换、重复值处理
3. 做 3 张图：分布图、分组对比、相关性/散点
4. 写结论：3 条发现 + 2 条下一步假设

## 交付物（你需要产出什么）
- 一个仓库（或文件夹）包含：
  - `README.md`：数据来源、运行方式、结论摘要
  - `requirements.txt`（或 `environment.yml`）
  - `notebooks/` 或 `src/`：可运行的分析代码

## 自检
- 换一台电脑/新建环境，能否按 README 一键跑通？
- 运行结果是否稳定（随机种子、版本固定）？
