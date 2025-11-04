"""
predict.py - Comprehensive Model Evaluation and Visualization (with TTA + threshold)
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Rangpur
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)
from tqdm import tqdm

from modules import build_model
from dataset import ADNIDataset

IDX_TO_CLASS = {0: "AD", 1: "NC"}
CLASS_TO_IDX = {"AD": 0, "NC": 1}


def tta_logits(model, x):
    """Simple TTA: average logits over original and horizontal flip."""
    logits_list = []
    with torch.no_grad():
        logits_list.append(model(x))
        logits_list.append(model(torch.flip(x, dims=[3])))  # H-flip
    return torch.stack(logits_list, dim=0).mean(dim=0)


def evaluate_model(model, loader, device):
    """
    Evaluate model and return predictions, labels, and probabilities (with TTA + threshold).
    Threshold is read from evaluate_model._thr (float).
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    thr = getattr(evaluate_model, "_thr", 0.5)
    
    with torch.no_grad():
        for x, y, paths in tqdm(loader, desc="Evaluating"):
            x = x.to(device)
            # TTA logits
            logits = tta_logits(model, x)
            probs = F.softmax(logits, dim=1).cpu().numpy()  # (B,2)
            preds = (probs[:, 1] >= thr).astype(np.int64)   # NC if P(NC) >= thr else AD
            
            # labels -> ints
            labels = [label.item() if torch.is_tensor(label) else label for label in y]
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs)
            all_paths.extend(paths)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs), all_paths


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['AD', 'NC'], yticklabels=['AD', 'NC'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix to {save_path}")


def plot_roc_curve(y_true, y_probs, save_path):
    # AUC for AD (class 0) as positive
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 0], pos_label=0)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve (AD positive)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ROC curve to {save_path}")
    return roc_auc


def plot_metrics_comparison(metrics, save_path):
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h, f'{h:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    plt.ylim([0, 1.1]); plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right'); plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved metrics comparison to {save_path}")


def plot_sample_predictions(dataset, predictions, labels, probs, save_path, num_samples=16):
    indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16)); axes = axes.flatten()
    for idx, ax in enumerate(axes):
        if idx >= len(indices):
            ax.axis('off'); continue
        i = indices[idx]
        img, true_label, _ = dataset[i]
        pred_label = int(predictions[i])
        prob = probs[i]
        true_label = true_label.item() if torch.is_tensor(true_label) else true_label
        img_np = img.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
        img_np = np.clip(std * img_np + mean, 0, 1)
        ax.imshow(img_np)
        true_class = IDX_TO_CLASS[true_label]; pred_class = IDX_TO_CLASS[pred_label]
        confidence = prob[pred_label] * 100
        color = 'green' if pred_label == true_label else 'red'
        ax.set_title(f'True: {true_class} | Pred: {pred_class}\nConf: {confidence:.1f}%',
                     color=color, fontsize=10, fontweight='bold')
        ax.axis('off')
    plt.suptitle('Sample Predictions', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"✓ Saved sample predictions to {save_path}")


def plot_misclassified(dataset, predictions, labels, probs, save_path, num_samples=16):
    mis_idx = np.where(predictions != labels)[0]
    if len(mis_idx) == 0:
        print("⚠ No misclassified samples found!"); return
    indices = np.random.choice(mis_idx, min(num_samples, len(mis_idx)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(16, 16)); axes = axes.flatten()
    for idx, ax in enumerate(axes):
        if idx >= len(indices):
            ax.axis('off'); continue
        i = indices[idx]
        img, true_label, _ = dataset[i]
        pred_label = int(predictions[i])
        prob = probs[i]
        true_label = true_label.item() if torch.is_tensor(true_label) else true_label
        img_np = img.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
        img_np = np.clip(std * img_np + mean, 0, 1)
        ax.imshow(img_np)
        true_class = IDX_TO_CLASS[true_label]; pred_class = IDX_TO_CLASS[pred_label]
        confidence = prob[pred_label] * 100
        ax.set_title(f'True: {true_class} | Pred: {pred_class}\nConf: {confidence:.1f}%',
                     color='red', fontsize=10, fontweight='bold')
        ax.axis('off')
    plt.suptitle('Misclassified Samples', fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"✓ Saved misclassified samples to {save_path}")


def plot_combined_confusion_roc(y_true, y_pred, y_probs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['AD', 'NC'], yticklabels=['AD', 'NC'],
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=11); ax1.set_xlabel('Predicted Label', fontsize=11)
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 0], pos_label=0)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax2.plot([0, 1], [0, 1], lw=2, linestyle='--', label='Random')
    ax2.set_xlim([0.0, 1.0]); ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=11)
    ax2.set_ylabel('True Positive Rate', fontsize=11)
    ax2.set_title('ROC Curve (AD positive)', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower right"); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"✓ Saved combined confusion matrix & ROC to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt model on ADNI test set")
    parser.add_argument("--checkpoint", type=str, default="runs/adni_convnext/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--data-root", type=str, default="/home/groups/comp3710/ADNI/AD_NC",
                        help="Path to ADNI dataset")
    parser.add_argument("--model", type=str, default="convnext_small",
                        help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--img-size", type=int, default=384,
                        help="Input image size")
    parser.add_argument("--save-dir", type=str, default="./images",
                        help="Directory to save visualizations")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="Decision threshold on P(NC). 0.5=argmax equivalent.")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*70)
    print("ConvNeXt ADNI Model Evaluation")
    print("="*70)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Save directory: {save_dir}")
    print(f"Threshold (P(NC)): {args.threshold}")
    print("="*70 + "\n")

    print("Loading test dataset...")
    test_set = ADNIDataset(args.data_root, split="test", img_size=args.img_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, 
                             shuffle=False, num_workers=args.num_workers, 
                             pin_memory=True)
    print(f"✓ Test samples: {len(test_set)}\n")

    print(f"Loading model...")
    model = build_model(args.model, num_classes=2, pretrained=False, drop_rate=0.2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model loaded successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}\n")

    # Set threshold for evaluator
    evaluate_model._thr = args.threshold

    print("Evaluating model on test set...")
    predictions, labels, probs, paths = evaluate_model(model, test_loader, device)
    print("✓ Evaluation complete\n")
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Specificity': specificity
    }
    
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Total Samples: {len(labels)}")
    print(f"Correct Predictions: {np.sum(predictions == labels)} ({np.sum(predictions == labels)/len(labels)*100:.2f}%)")
    print(f"Incorrect Predictions: {np.sum(predictions != labels)} ({np.sum(predictions != labels)/len(labels)*100:.2f}%)")
    print()
    print(f"Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    print("\n" + "-"*70)
    print("Per-Class Metrics:")
    print("-"*70)
    print(classification_report(labels, predictions, target_names=['AD', 'NC'], digits=4))
    
    print("Confusion Matrix Breakdown:")
    print(f"  True Negatives  (AD correctly classified): {cm[0,0]:>5d}")
    print(f"  False Positives (AD misclassified as NC):  {cm[0,1]:>5d}")
    print(f"  False Negatives (NC misclassified as AD):  {cm[1,0]:>5d}")
    print(f"  True Positives  (NC correctly classified): {cm[1,1]:>5d}")
    print("="*70 + "\n")
    
    print("Generating visualizations...")
    print("-"*70)
    plot_confusion_matrix(labels, predictions, save_dir / "confusion_matrix.png")
    roc_auc = plot_roc_curve(labels, probs, save_dir / "roc_curve.png")
    metrics['AUC-ROC'] = roc_auc
    plot_combined_confusion_roc(labels, predictions, probs, save_dir / "confusion_matrix_roc.png")
    plot_metrics_comparison(metrics, save_dir / "performance_metrics.png")
    plot_sample_predictions(test_set, predictions, labels, probs, save_dir / "sample_predictions.png", num_samples=16)
    plot_misclassified(test_set, predictions, labels, probs, save_dir / "misclassified_samples.png", num_samples=16)
    
    metrics_path = save_dir / "metrics.json"
    metrics_dict = {k: float(v) for k, v in metrics.items()}
    metrics_dict['total_samples'] = int(len(labels))
    metrics_dict['correct_predictions'] = int(np.sum(predictions == labels))
    metrics_dict['incorrect_predictions'] = int(np.sum(predictions != labels))
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"✓ Saved metrics to {metrics_path}")
    
    print("-"*70)
    print(f"\n✅ All visualizations saved to: {save_dir.absolute()}/")
    print("\nGenerated files:")
    print("  • confusion_matrix.png")
    print("  • roc_curve.png")
    print("  • confusion_matrix_roc.png (combined)")
    print("  • performance_metrics.png")
    print("  • sample_predictions.png")
    print("  • misclassified_samples.png")
    print("  • metrics.json")
    print(f"\nscp -r s4977354@rangpur.rcc.uq.edu.au:{save_dir.absolute()} ./")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

