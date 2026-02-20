# 附录 05：常见报错与排错路线图

## 1) 先定位是“数据问题”还是“代码问题”
- 如果训练脚本直接报错：优先是代码/依赖问题
- 如果能跑但指标很怪：优先怀疑数据/切分/泄漏

## 2) 三件套（永远先做）
- 打印 `X.shape`, `y.shape`
- 打印 `X.dtypes`
- 打印 `X.head()` / `y.value_counts()`

## 3) 指标异常好（AUC 0.99）
- 高概率是泄漏：看附录 02
- 做“打乱标签”实验验证

## 4) 指标接近随机
- 先做基线：多数类/均值预测
- 检查标签是否对齐（merge 后错位最常见）
- 检查特征是否全是常数/全缺失

## 5) sklearn 常见报错
- `could not convert string to float`：类别列没编码
- `Found input variables with inconsistent numbers of samples`：X/y 对齐错
- `Unknown label type`：分类标签格式不对

## 6) PyTorch 常见报错
- `Expected all tensors to be on the same device`：device 不一致
- `CUDA out of memory`：batch 太大/模型太大/没清理
- loss 变成 NaN：学习率过大、输入异常、除零

## 7) 环境/命令常见报错（macOS / zsh）
- `zsh: command not found: conda`
	- 含义：系统找不到 conda（没装，或 PATH 没生效）
	- 最快解决：别卡在 conda，上教程的 **venv + pip** 方案先跑通
	- 如果你确实要用 conda：安装 Miniconda/Anaconda 后执行 `conda init zsh`，重开终端再试 `conda --version`
