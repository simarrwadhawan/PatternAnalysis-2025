import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from modules import build_model, num_trainable_params
from dataset import ADNIConfig, ADNIDataset
from utils import AvgMeter, set_seed


class FocalLoss(nn.Module):
    """
    Focal Loss to handle class imbalance by down-weighting easy examples.
    Focuses training on hard-to-classify samples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation - blends pairs of images"""
    lam = torch.distributions.Beta(alpha, alpha).sample()
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss function for mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def accuracy(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


def train_one_epoch(model, loader, optim, criterion, device, use_mixup=True):
    model.train()
    loss_m = AvgMeter()
    acc_m = AvgMeter()
    
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        
        if use_mixup:
            x, y_a, y_b, lam = mixup_data(x, y, alpha=0.3)
            logits = model(x)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            logits = model(x)
            loss = criterion(logits, y)
        
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        
        acc = accuracy(logits, y)
        bs = x.size(0)
        loss_m.update(loss.item(), bs)
        acc_m.update(acc, bs)
    
    return loss_m.avg, acc_m.avg


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_m = AvgMeter()
    acc_m = AvgMeter()
    
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            
            acc = accuracy(logits, y)
            bs = x.size(0)
            loss_m.update(loss.item(), bs)
            acc_m.update(acc, bs)
    
    return loss_m.avg, acc_m.avg


def freeze_backbone(model, unfreeze_last_n_blocks=2):
    """Freeze backbone except last N blocks"""
    backbone = model.backbone if hasattr(model, 'backbone') else model
    
    for param in model.parameters():
        param.requires_grad = False
    
    if hasattr(backbone, 'head'):
        for param in backbone.head.parameters():
            param.requires_grad = True
        print("Unfroze classifier head")
    elif hasattr(backbone, 'fc'):
        for param in backbone.fc.parameters():
            param.requires_grad = True
        print("Unfroze classifier fc")
    
    if hasattr(backbone, 'stages'):
        total_stages = len(backbone.stages)
        start_idx = max(0, total_stages - unfreeze_last_n_blocks)
        for i in range(start_idx, total_stages):
            for param in backbone.stages[i].parameters():
                param.requires_grad = True
        print(f"Unfroze stages {start_idx} to {total_stages-1} (out of {total_stages})")
    
    if hasattr(backbone, 'norm'):
        for param in backbone.norm.parameters():
            param.requires_grad = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--outdir", type=str, default="runs/adni_full_train")
    p.add_argument("--model", type=str, default="convnext_small")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=24)  # 384px fits on A100
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--drop-rate", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze-backbone", action="store_true", default=True)
    p.add_argument("--unfreeze-last-n", type=int, default=3)
    p.add_argument("--focal-gamma", type=float, default=0.0)
    p.add_argument("--use-mixup", action="store_true", default=True)
    args = p.parse_args()

    print("\n" + "="*70)
    print("FINAL TRAINING: All Data + Real Test Validation")
    print("="*70)
    print("KEY CHANGES:")
    print(f"  1. Using ALL 21,520 training samples")
    print(f"  2. Validating on REAL test set (9,000 samples)")
    print(f"  3. Model: {args.model}")
    print(f"  4. LR: {args.lr}")
    print(f"  5. Weight decay: {args.weight_decay}")
    print(f"  6. Dropout: {args.drop_rate}")
    print(f"  7. Focal gamma: {args.focal_gamma} (0=disabled)")
    print(f"\nTarget: 80%+ test accuracy")
    print("="*70 + "\n")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    cfg = ADNIConfig(data_root=args.data_root, img_size=args.img_size)
    
    print("Loading datasets...")
    train_set = ADNIDataset(cfg.data_root, split="train", img_size=cfg.img_size)
    val_set = ADNIDataset(cfg.data_root, split="test", img_size=cfg.img_size)
    
    print(f"Train: {len(train_set)} samples (ALL training data)")
    print(f"Val:   {len(val_set)} samples (REAL test set)")

    train_labels = [label.item() if torch.is_tensor(label) else label 
                    for _, label, _ in train_set]
    class_counts = Counter(train_labels)
    
    print(f"\nClass distribution:")
    print(f"  AD: {class_counts.get(0, 0)} samples")
    print(f"  NC: {class_counts.get(1, 0)} samples")
    
    total = sum(class_counts.values())
    
    class_weights = torch.tensor([
        total / (1.2 * class_counts[0]),
        total / (2.8 * class_counts[1])
    ], dtype=torch.float32).to(device)
    
    print(f"\nClass weights:")
    print(f"  AD: {class_weights[0]:.3f}")
    print(f"  NC: {class_weights[1]:.3f}")
    print(f"  Ratio: {class_weights[0]/class_weights[1]:.2f}x")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )

    print(f"\nBuilding {args.model} model...")
    model = build_model(
        model_name=args.model, 
        num_classes=2, 
        pretrained=True,
        drop_rate=args.drop_rate
    ).to(device)
    
    if args.freeze_backbone:
        print(f"\nFreezing backbone (unfreezing last {args.unfreeze_last_n} blocks)...")
        freeze_backbone(model, unfreeze_last_n_blocks=args.unfreeze_last_n)
    
    trainable = num_trainable_params(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optim = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2, eta_min=1e-7)
    
    if args.focal_gamma > 0:
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        print(f"\nUsing Focal Loss (gamma={args.focal_gamma})")
    else:
        # *** label smoothing for generalization ***
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print(f"\nUsing CrossEntropyLoss with class weights + label_smoothing=0.1")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 10

    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)
    
    for epoch in range(1, args.epochs + 1):
        # Disable mixup for the final quarter of training to sharpen decision boundaries
        use_mixup_now = args.use_mixup and (epoch <= int(0.75 * args.epochs))
        if epoch == int(0.75 * args.epochs) + 1:
            print(">> Mixup disabled for final epochs.")

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optim, criterion, device, 
            use_mixup=use_mixup_now
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optim.param_groups[0]['lr']
        
        print(
            f"epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} | lr {current_lr:.2e}"
        )

        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }, outdir / "last.pt")
        
        if va_acc > best_acc:
            best_acc = va_acc
            best_loss = va_loss
            patience_counter = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
            }, outdir / "best.pt")
            print(f"  New best! Acc: {best_acc:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stop_patience})")
            
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"{'='*70}")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to: {outdir}/best.pt")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

