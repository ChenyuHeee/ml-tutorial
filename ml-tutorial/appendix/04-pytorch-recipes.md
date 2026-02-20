# 附录 04：PyTorch 实战配方

## 1) 设备选择（macOS 重点）
- 有 NVIDIA GPU：优先 `cuda`
- Apple Silicon：可能可用 `mps`
- 其它：`cpu`

训练时一定要保证：**模型与 batch 在同一个 device**。

## 2) 训练循环最小清单
- `model.train()`（训练）
- `optimizer.zero_grad()`
- forward → loss → `loss.backward()` → `optimizer.step()`
- 评估：`model.eval()` + `torch.no_grad()`

## 3) 常见不收敛原因（按概率从高到低）
- 学习率太大/太小
- 输入没有归一化（例如图像没做 ToTensor/Normalize）
- 标签编码不对（CrossEntropy 需要类别 id，而不是 one-hot）
- 忘了 `model.train()` / `model.eval()`

## 4) 过拟合/欠拟合怎么判断
- 过拟合：train acc 上升，val acc 停滞或下降
  - 解决：数据增强、dropout、weight decay、更小模型、早停
- 欠拟合：train acc 也上不去
  - 解决：更大模型、训练更久、调学习率、检查数据/标签

## 5) 推荐你先抄骨架再改
- `ml-tutorial/exercises/pytorch_train_skeleton.py`
