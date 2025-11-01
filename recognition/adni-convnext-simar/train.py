import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from modules import build_model
from dataset import ADNIConfig, make_splits
from utils import AvgMeter, set_seed


def accuracy(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader, optim, device):
    model.train()
    loss_m = AvgMeter()
    acc_m = AvgMeter()
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        acc = accuracy(logits, y)
        bs = x.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update(acc, bs)
    return loss_m.avg, acc_m.avg


def evaluate(model, loader, device):
    model.eval()
    loss_m = AvgMeter()
    acc_m = AvgMeter()
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            acc = accuracy(logits, y)
            bs = x.size(0)
            loss_m.update(loss.item(), bs)
            acc_m.update(acc, bs)
    return loss_m.avg, acc_m.avg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--outdir", type=str, default="runs/adni_convnext")
    p.add_argument("--model", type=str, default="convnext_tiny")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = ADNIConfig(data_root=args.data_root, img_size=args.img_size)
    train_set, val_set = make_splits(cfg, seed=args.seed)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    model = build_model(model_name=args.model, num_classes=2, pretrained=True).to(device)
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, device)
        va_loss, va_acc = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )

        torch.save({"model": model.state_dict()}, outdir / "last.pt")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict()}, outdir / "best.pt")

    print(f"best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
