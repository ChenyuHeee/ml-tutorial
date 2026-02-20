# 从零开始的机器学习教程（可跟读 + 可动手）

这是一套“你照着做就能学会”的教程，而不是概览式大纲。

## 适合谁
- 计算机专业学生
- 会基础编程（Python/Java/C++ 任何一个都行），但机器学习从零

## 你需要的学习方式（很重要）
- 每章：先读“概念最小集”→ 再做“必做练习”→ 最后完成“验收任务”
- 遇到不懂：先做错误分析/打印 shape/看指标，再回到概念

## 进度建议
- 轻量：每周 6–8 小时（12–16 周）
- 标准：每周 10–12 小时（10–12 周）

## 教程路线（按顺序完成）
0. 环境准备与工作方式 → [chapters/00-setup.md](chapters/00-setup.md)
1. Python 数据处理入门（Numpy/Pandas）→ [chapters/01-python-data.md](chapters/01-python-data.md)
2. 机器学习问题是什么：数据、标签、指标、验证 → [chapters/02-ml-basics.md](chapters/02-ml-basics.md)
3. 线性回归与第一个可复现训练脚本 → [chapters/03-linear-regression.md](chapters/03-linear-regression.md)
4. 逻辑回归与二分类评估（含阈值）→ [chapters/04-logistic-regression.md](chapters/04-logistic-regression.md)
5. 特征工程与数据泄漏（上手 sklearn Pipeline）→ [chapters/05-features-and-leakage.md](chapters/05-features-and-leakage.md)
6. 树模型与交叉验证（RandomForest/GBDT）→ [chapters/06-trees-and-cv.md](chapters/06-trees-and-cv.md)
7. 无监督：聚类与降维可视化 → [chapters/07-unsupervised.md](chapters/07-unsupervised.md)
8. PyTorch 入门：从零训练一个小网络 → [chapters/08-pytorch-basics.md](chapters/08-pytorch-basics.md)
9. 做一个作品集级小项目（端到端）→ [projects/capstone.md](projects/capstone.md)

## 附录（速查 + 排错 + 配方）
学习/做项目时随时查：指标与验证、数据泄漏、sklearn/PyTorch 配方与常见报错。
- 附录总览 → [appendix/README.md](appendix/README.md)

## 作业与模板
- 每周打卡模板 → [templates/weekly-checklist.md](templates/weekly-checklist.md)
- 项目 README 模板（保证可复现）→ [templates/project-readme-template.md](templates/project-readme-template.md)

## 代码脚手架（可直接复制）
如果你不知道每章的 `train.py/evaluate.py` 怎么组织，可以先复制本教程提供的最小脚手架，再按章节要求改：
- 代码模板目录 → [exercises/README.md](exercises/README.md)

## 你的练习代码放哪里
建议你在工作区新建一个练习目录（你也可以让我帮你生成骨架）：
- `ml-work/`：每章一个子文件夹（例如 `ch03-linear-reg/`）

> 提示：本教程的目标是让你“能独立做项目”。所以每章都要求你写脚本、固定随机种子、输出指标，并做简单误差分析。
