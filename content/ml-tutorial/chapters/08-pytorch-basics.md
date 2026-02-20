# 第 8 章：PyTorch 入门（1–2 周）

目标：你能用 PyTorch 训练一个小模型，掌握训练循环、GPU/CPU、过拟合诊断。

## 8.1 安装说明（重要）
PyTorch 的安装会因 CPU/GPU 与 CUDA 版本不同而不同：
- 请优先按官网安装命令安装
- 如果你先只做 CPU 也完全可以

## 8.2 概念最小集
- Tensor 与自动求导（autograd）
- Dataset/DataLoader
- 模型：`nn.Module`
- 损失与优化：`nn.CrossEntropyLoss`、`torch.optim.Adam`
- 训练循环：train/eval 模式切换

你必须掌握的训练循环最小清单：
- 选择 `device`（cpu / cuda / mps）并把模型与 batch 移过去
- `model.train()` 与 `model.eval()` 切换（评估时关掉 dropout 等）
- 用 `torch.no_grad()` 包住评估
- 每个 epoch 输出 loss/acc（至少在 train 与 val 各 1 组）

## 8.3 必做练习
- 用一个小数据集做分类（MNIST 或 CIFAR-10 都可以）
- 实现：
  - `train.py`：训练、打印 loss/acc
  - `evaluate.py`：在验证集评估
- 做 2 次对比实验：
  1) 不加正则（容易过拟合）
  2) 加 dropout 或 weight decay（看曲线变化）

建议你把每次实验的配置写到 README：
- 模型结构（层数、隐藏维度、是否 dropout）
- optimizer 与学习率
- batch size 与 epoch

## 8.4 验收任务
在 `ml-work/ch08-pytorch/` 里完成：
- 训练曲线截图或日志（至少 10 个 epoch）
- 写 5 行结论：你观察到的过拟合/欠拟合迹象是什么？你怎么改？

验收标准：
- 你能提供一段训练日志（或一张曲线图），看得出 train 与 val 的差距变化
- 你能解释：你改的正则/超参为什么可能有效

完成本章后进入最终项目。

---

下面把“能跑通 + 能解释 + 能复现”的 PyTorch 训练闭环直接给你。

## 8.5 环境自检（1 分钟）
PyTorch 安装命令随 CPU/CUDA 不同而变，优先按官方安装页来。

安装完成后跑：

```bash
python -c "import torch; print(torch.__version__); print('cuda=', torch.cuda.is_available()); print('mps=', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"


你在 macOS 上如果是 Apple Silicon，常见是 `mps=True`（表示可以用 GPU 加速）。

## 8.6 直接可用：`train.py`（MNIST 最小训练循环 + 保存 best）
在 `ml-work/ch08-pytorch/` 新建 `train.py`，运行：

```bash
python train.py
```

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class Config:
  batch_size: int = 128
  lr: float = 1e-3
  epochs: int = 10
  seed: int = 42
  num_workers: int = 2
  out_dir: str = "artifacts"
  # 正则：你可以把这里改成 0.0 做对比实验
  weight_decay: float = 1e-4
  dropout: float = 0.2


def pick_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  # Apple Silicon
  if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


class MLP(nn.Module):
  def __init__(self, dropout: float) -> None:
    super().__init__()
    self.fc1 = nn.Linear(28 * 28, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 10)
    self.drop = nn.Dropout(p=dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = self.drop(x)
    x = F.relu(self.fc2(x))
    x = self.drop(x)
    return self.fc3(x)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
  preds = logits.argmax(dim=1)
  return float((preds == y).float().mean().item())


def train_one_epoch(
  model: nn.Module,
  loader: DataLoader,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
) -> tuple[float, float]:
  model.train()
  total_loss = 0.0
  total_acc = 0.0
  n_batches = 0

  for x, y in loader:
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

    total_loss += float(loss.item())
    total_acc += accuracy(logits.detach(), y)
    n_batches += 1

  return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


@torch.no_grad()
def evaluate(
  model: nn.Module,
  loader: DataLoader,
  device: torch.device,
) -> tuple[float, float]:
  model.eval()
  total_loss = 0.0
  total_acc = 0.0
  n_batches = 0

  for x, y in loader:
    x = x.to(device)
    y = y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits, y)

    total_loss += float(loss.item())
    total_acc += accuracy(logits, y)
    n_batches += 1

  return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def main() -> None:
  cfg = Config()
  torch.manual_seed(cfg.seed)

  device = pick_device()
  print("device:", device)

  tfm = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)),
    ]
  )

  train_full = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
  test_ds = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

  # 切分 train/val（固定 seed 保证可复现）
  n_train = int(0.9 * len(train_full))
  n_val = len(train_full) - n_train
  train_ds, val_ds = torch.utils.data.random_split(
    train_full,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(cfg.seed),
  )

  train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
  )
  val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
  )
  test_loader = DataLoader(
    test_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
  )

  model = MLP(dropout=cfg.dropout).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(exist_ok=True)
  ckpt_path = out_dir / "mnist_mlp.pt"

  best_val_acc = -1.0
  for epoch in range(1, cfg.epochs + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
    va_loss, va_acc = evaluate(model, val_loader, device)
    print(
      f"epoch={epoch:02d} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
      f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
    )

    if va_acc > best_val_acc:
      best_val_acc = va_acc
      torch.save(
        {
          "model_state_dict": model.state_dict(),
          "config": cfg.__dict__,
        },
        ckpt_path,
      )

  print("saved:", ckpt_path.as_posix())

  # 顺手用当前权重看一下 test（严格做法：evaluate.py 加载 best）
  te_loss, te_acc = evaluate(model, test_loader, device)
  print(f"test_loss={te_loss:.4f} test_acc={te_acc:.4f}")


if __name__ == "__main__":
  main()
```

预期现象（范围即可，别纠结具体数字）：
- 10 个 epoch 后 `val_acc` 往往能到 `0.96+`
- `train_acc` 通常略高于 `val_acc`（差距明显变大就要考虑过拟合）

## 8.7 直接可用：`evaluate.py`（加载 best checkpoint 做评估）
新建 `evaluate.py`，运行：

```bash
python evaluate.py
```

```python
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def pick_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


class MLP(nn.Module):
  def __init__(self, dropout: float) -> None:
    super().__init__()
    self.fc1 = nn.Linear(28 * 28, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 10)
    self.drop = nn.Dropout(p=dropout)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = self.drop(x)
    x = F.relu(self.fc2(x))
    x = self.drop(x)
    return self.fc3(x)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
  preds = logits.argmax(dim=1)
  return float((preds == y).float().mean().item())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
  model.eval()
  total_loss = 0.0
  total_acc = 0.0
  n_batches = 0

  for x, y in loader:
    x = x.to(device)
    y = y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    total_loss += float(loss.item())
    total_acc += accuracy(logits, y)
    n_batches += 1

  return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def main() -> None:
  device = pick_device()
  print("device:", device)

  ckpt_path = Path("artifacts") / "mnist_mlp.pt"
  ckpt = torch.load(ckpt_path, map_location=device)
  cfg = ckpt.get("config", {})

  model = MLP(dropout=float(cfg.get("dropout", 0.0))).to(device)
  model.load_state_dict(ckpt["model_state_dict"])

  tfm = transforms.Compose(
    [
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)),
    ]
  )
  test_ds = datasets.MNIST(root="data", train=False, download=True, transform=tfm)
  test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

  te_loss, te_acc = evaluate(model, test_loader, device)
  print(f"test_loss={te_loss:.4f} test_acc={te_acc:.4f}")


if __name__ == "__main__":
  main()
```

## 8.8 你必须能说清的“过拟合诊断”
当你看到：
- `train_acc` 持续上升，但 `val_acc` 停滞甚至下降
- `train_loss` 下降，但 `val_loss` 上升

通常是过拟合。你可以优先试两类操作：
- 更强正则：增大 `weight_decay`、增大 `dropout`、早停
- 降低模型容量：减小隐藏层维度/层数

## 8.9 验收清单（全部满足）
- `python train.py` 能训练 10 epoch，并保存 `artifacts/mnist_mlp.pt`
- `python evaluate.py` 能加载并输出测试集 `test_acc`
- 你做两次对比：`weight_decay/dropout` 为 `0` vs 非 `0`，并写 5 行结论
