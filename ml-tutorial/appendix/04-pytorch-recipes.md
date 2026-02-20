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
下面是一个“你可以直接复制”的最小训练骨架（分类/回归都能改）：

```python
from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


@dataclass
class Config:
  batch_size: int = 128
  lr: float = 1e-3
  epochs: int = 10


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: torch.device) -> float:
  model.train()
  total_loss = 0.0
  n = 0
  for xb, yb in loader:
    xb = xb.to(device)
    yb = yb.to(device)

    optimizer.zero_grad(set_to_none=True)
    pred = model(xb)
    loss = loss_fn(pred.squeeze(), yb)
    loss.backward()
    optimizer.step()

    bs = xb.shape[0]
    total_loss += loss.item() * bs
    n += bs
  return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
  model.eval()
  total_loss = 0.0
  n = 0
  for xb, yb in loader:
    xb = xb.to(device)
    yb = yb.to(device)
    pred = model(xb)
    loss = loss_fn(pred.squeeze(), yb)
    bs = xb.shape[0]
    total_loss += loss.item() * bs
    n += bs
  return total_loss / max(n, 1)


def main() -> None:
  cfg = Config()
  device = get_device()
  print("device:", device)

  # TODO: 用你的数据替换这里（示例：随机数据）
  x = torch.randn(2048, 20)
  y = torch.randn(2048)  # 回归示例；分类时你可以改成 0/1 并用 BCEWithLogitsLoss

  ds = TensorDataset(x, y)
  train_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
  val_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)

  model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
  ).to(device)

  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
  loss_fn = nn.MSELoss()

  for epoch in range(1, cfg.epochs + 1):
    t0 = time.time()
    train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
    val_loss = evaluate(model, val_loader, loss_fn, device)
    dt = time.time() - t0
    print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {dt:.1f}s")


if __name__ == "__main__":
  main()
```
