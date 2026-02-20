import argparse
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass(frozen=True)
class Config:
    batch_size: int
    epochs: int
    lr: float
    seed: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return Config(batch_size=args.batch_size, epochs=args.epochs, lr=args.lr, seed=args.seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Apple Silicon
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SmallNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def main() -> None:
    cfg = parse_args()

    torch.manual_seed(cfg.seed)

    device = get_device()
    print(f"device={device}")

    transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    model = SmallNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, test_loader, device)
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
