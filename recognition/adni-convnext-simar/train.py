import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from modules import build_model, num_trainable_params
from dataset import ADNIConfig, make_splits
from utils import AvgMeter, set_seed


def accuracy(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader, optim, device, class_weights):
    model.train()
    loss_m = AvgMeter()
    acc_m = AvgMeter()
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, weight=class_weights, label_smoothing=0.1)
        
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def freeze_backbone(model, unfreeze_last_n_blocks=1):
    """Freeze backbone except last N blocks - works with ConvNeXtClassifier wrapper"""
    # Get the actual timm model (it's wrapped in model.backbone)
    backbone = model.backbone if hasattr(model, 'backbone') else model
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the classifier head
    if hasattr(backbone, 'head'):
        for param in backbone.head.parameters():
            param.requires_grad = True
        print(f"✓ Unfroze classifier head")
    elif hasattr(backbone, 'fc'):
        for param in backbone.fc.parameters():
            param.requires_grad = True
        print(f"✓ Unfroze classifier fc")
    else:
        print("⚠ Warning: Could not find classifier head!")
        return
    
    # Unfreeze last N stages of ConvNeXt
    if hasattr(backbone, 'stages'):
        total_stages = len(backbone.stages)
        start_idx = max(0, total_stages - unfreeze_last_n_blocks)
        for i in range(start_idx, total_stages):
            for param in backbone.stages[i].parameters():
                param.requires_grad = True
        print(f"✓ Unfroze stages {start_idx} to {total_stages-1} (out of {total_stages})")
    else:
        print("⚠ Warning: Could not find backbone stages!")
    
    # Unfreeze norm layer
    if hasattr(backbone, 'norm'):
        for param in backbone.norm.parameters():
            param.requires_grad = True
        print("✓ Unfroze norm layer")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--outdir", type=str, default="runs/adni_convnext")
    p.add_argument("--model", type=str, default="convnext_tiny")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=12)  # Reduced from 16
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-4)  # Increased from 5e-5
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--drop-rate", type=float, default=0.2)  # Reduced from 0.3
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze-backbone", action="store_true", default=True)
    p.add_argument("--unfreeze-last-n", type=int, default=2)  # Increased from 1
    args = p.parse_args()

    print("Starting training...")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets using make_splits for proper train/val split
    cfg = ADNIConfig(data_root=args.data_root, img_size=args.img_size, train_ratio=0.8)
    train_set, val_set = make_splits(cfg, seed=args.seed)

    # Calculate class weights to handle imbalance
    # Convert tensor labels to integers for Counter
    train_labels = [label.item() if torch.is_tensor(label) else label for _, label, _ in train_set]
    class_counts = Counter(train_labels)
    
    print(f"\nClass distribution:")
    print(f"  Class 0 (AD): {class_counts.get(0, 0)} samples")
    print(f"  Class 1 (NC): {class_counts.get(1, 0)} samples")
    
    # Check if we have both classes
    if len(class_counts) < 2:
        print(f"\n⚠️  WARNING: Only {len(class_counts)} class found in training data!")
        print(f"Classes present: {list(class_counts.keys())}")
        raise ValueError("Training data must contain both AD and NC samples")
    
    total = sum(class_counts.values())
    class_weights = torch.tensor([
        total / (2 * class_counts[0]),  # Weight for AD (class 0)
        total / (2 * class_counts[1])   # Weight for NC (class 1)
    ], dtype=torch.float32).to(device)
    
    print(f"Class weights: AD={class_weights[0]:.3f}, NC={class_weights[1]:.3f}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    print(f"\nTrain samples: {len(train_set)}")
    print(f"Val samples: {len(val_set)}")

    # Build model
    model = build_model(
        model_name=args.model, 
        num_classes=2, 
        pretrained=True,
        drop_rate=args.drop_rate
    ).to(device)
    
    # Freeze backbone for transfer learning
    if args.freeze_backbone:
        print(f"\nFreezing backbone, unfreezing last {args.unfreeze_last_n} block(s)...")
        freeze_backbone(model, unfreeze_last_n_blocks=args.unfreeze_last_n)
    
    trainable = num_trainable_params(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")
    
    # Get trainable parameters only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Optimizer with higher weight decay
    optim = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Better scheduler: Cosine annealing with restarts
    scheduler = CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, eta_min=1e-7)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = 10

    print("\nStarting training loop...")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optim, device, class_weights)
        va_loss, va_acc = evaluate(model, val_loader, device)
        
        # Step scheduler every epoch
        scheduler.step()
        current_lr = optim.param_groups[0]['lr']
        
        print(
            f"epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {current_lr:.2e}"
        )

        # Save last model
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "optimizer": optim.state_dict(),
        }, outdir / "last.pt")
        
        # Save best model
        if va_acc > best_acc:
            best_acc = va_acc
            patience_counter = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
            }, outdir / "best.pt")
            print(f"  ✓ New best model! Accuracy: {best_acc:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
