# 第 1 章：Python 数据处理入门（Numpy/Pandas）（1 周）

目标：你能独立完成一次数据加载、清洗、统计与可视化，并写出“可复现”的分析。

## 1.1 概念最小集
- Numpy：数组、广播、向量化（尽量少写 Python for 循环）
- Pandas：DataFrame、索引、过滤、groupby、merge、缺失值处理
- 可视化：用图回答问题（分布、对比、趋势、关系）

## 1.2 必做练习（建议放到 `ml-work/ch01-python-data/`）
选一个公开数据集（CSV 即可）。为了避免卡在下载/注册上，优先选“无需账号就能拿到”的：
- 方案 A（推荐，最省事）：直接用 pandas 从公开 URL 读取（例如 seaborn 的 titanic 数据）
- 方案 B：用你自己已有的 CSV（学校作业/爬虫/日志导出）
- 方案 C：用 sklearn 自带数据集（没有 CSV 也能做分析）

你可以直接用下面这个（无需注册）：
- `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv`

### 快速开始：把“必做练习”跑通（建议你先照抄）
在 `ml-work/ch01-python-data/` 新建一个 `eda.ipynb`，先运行下面这段：

```python
import pandas as pd

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

print("shape:", df.shape)
print("columns:", list(df.columns))
print("dtypes:\n", df.dtypes)

# 缺失率 top 10
missing_rate = df.isna().mean().sort_values(ascending=False)
print("\nmissing_rate_top10:\n", missing_rate.head(10))

# 目标列先看一眼（这是后面第 2 章会用到的监督学习例子）
print("\nsurvived value_counts:\n", df["survived"].value_counts())
```

你应当能看到：
- `age / deck / embark_town` 等列有较高缺失率
- `survived` 大致是 0/1 的二分类

完成以下步骤（按顺序，每一步都要把结果写到 notebook 或脚本里）：
1. 读取数据：打印行列数、列类型、缺失率 top 10
2. 清洗：
   - 修正明显的 dtype（例如字符串数字）
   - 缺失值策略（删除/填充）并记录理由
3. 探索性分析（EDA）：至少 3 个问题 + 对应图
   - 例如：某个数值特征的分布？不同类别的对比？两变量相关？
4. 输出一个干净数据：保存为 `clean.csv` 或 `clean.parquet`

### 3 个“你可以直接用”的 EDA 问题（以 Titanic 为例）
你不需要想很复杂的问题，先练“把问题 → 图 → 结论”这条链路：
1. **年龄分布如何？幸存者 vs 罹难者是否不同？**（直方图/核密度图 + 分组）
2. **不同舱位（pclass）幸存率是否不同？**（柱状图：均值/比例）
3. **票价（fare）与幸存是否有关？**（箱线图/分位数统计）

示例代码（你可以直接复制到 notebook）：

```python
import matplotlib.pyplot as plt

ax = df["age"].dropna().plot(kind="hist", bins=30, title="Age distribution")
ax.set_xlabel("age")
plt.show()

survival_by_class = df.groupby("pclass")["survived"].mean().sort_index()
ax = survival_by_class.plot(kind="bar", title="Survival rate by pclass")
ax.set_xlabel("pclass")
ax.set_ylabel("survival rate")
plt.show()

ax = df.boxplot(column="fare", by="survived")
plt.suptitle("")
plt.title("Fare by survived")
plt.show()
```

## 1.3 本章你需要产出哪些文件（照着做）
在 `ml-work/ch01-python-data/` 下：
- `eda.ipynb`：你的探索与图表
- `export_clean.py`（可选但推荐）：把清洗流程脚本化并导出 `clean.csv`
- `README.md`：用 10 行以内写清楚你的结论

### `export_clean.py` 最小模板（推荐你本章就写）
目的：你以后做 ML 时，不会因为 notebook 手工操作导致“跑不回去”。

```python
from pathlib import Path
import pandas as pd


def main() -> None:
   url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
   df = pd.read_csv(url)

   # 例子：保留后面做监督学习最常用的一些列（你可以按需要改）
   keep = ["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
   df = df[keep]

   # 简单缺失值处理（只是示例：你需要在 README 里解释为什么这么做）
   df["age"] = df["age"].fillna(df["age"].median())
   df["embarked"] = df["embarked"].fillna("UNK")

   out_dir = Path(__file__).resolve().parent
   out_path = out_dir / "clean.csv"
   df.to_csv(out_path, index=False)
   print("saved:", out_path)
   print("shape:", df.shape)


if __name__ == "__main__":
   main()
```

验收时，你应该能在终端运行：
- `python export_clean.py`
并看到它保存 `clean.csv`。

## 1.4 验收任务
- 你在 `README.md` 里写清楚：
  - 数据来源
  - 你提出的 3 个问题
  - 你用图得到的 3 个结论
  - 你做了哪些清洗（含缺失值策略）

最低标准（务必达成）：
- `eda.ipynb` 能从头运行到尾（Kernel 重启后也能跑通）
- `clean.csv` 是你“解释得清楚”的版本：每个清洗动作都有理由

额外加分（不强制）：
- 把“清洗前 vs 清洗后”的行数、缺失率变化用表格写出来

## 1.5 常见坑
- 把缺失值随便填 0：会引入强烈的假信号
- 用全量数据先做统计再划分：后面做 ML 会变成泄漏
- 图太多但没结论：每张图必须回答一个问题

再补两个高频坑：
- **把类别当数值**（例如 `sex` 直接留字符串，后面模型会报错）：你需要 one-hot 或编码（第 5 章会系统解决）。
- **做了清洗但没记录**：你自己一周后就忘；所以本章强制写 README。

完成本章后进入第 2 章。
