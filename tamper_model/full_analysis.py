"""
Comprehensive Model Analysis & Testing Script
==============================================
Generates a full evaluation report with multiple graphs and metrics:
  1. Confusion matrices (segmentation + classification)
  2. ROC curves (both tasks)
  3. Precision-Recall curves
  4. Sample predictions with masks overlay
  5. Per-class metrics breakdown
  6. Threshold analysis
  7. Error distribution analysis
  8. Per-image segmentation quality
  9. Summary bar chart
All plots saved to evaluation_report/
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from model import build_model, FrequencyPreprocessLayer
from dataloader import get_datasets, IMG_SIZE
from losses import dice_coefficient, segmentation_loss, classification_loss
from bayerconv_layer import BayarConv
from texture_stream import LocalAttentionLayer

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, accuracy_score
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "manifests")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
REPORT_DIR = os.path.join(PROJECT_ROOT, "evaluation_report")
os.makedirs(REPORT_DIR, exist_ok=True)

BATCH_SIZE = 32
THRESHOLD = 0.5


def load_model_and_data():
    print("Loading best model...")
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.keras")
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'segmentation_loss': segmentation_loss,
        'classification_loss': classification_loss,
        'BayarConv': BayarConv,
        'LocalAttentionLayer': LocalAttentionLayer,
        'FrequencyPreprocessLayer': FrequencyPreprocessLayer,
    }
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"  Loaded from {model_path}")

    print("Loading test dataset...")
    _, _, test_dataset = get_datasets(
        manifest_dir=MANIFEST_DIR,
        batch_size=BATCH_SIZE,
        project_root=PROJECT_ROOT,
    )
    return model, test_dataset


def collect_predictions(model, test_dataset):
    seg_true, seg_prob = [], []
    cls_true, cls_prob = [], []
    sample_images = []

    print("Running inference on test set...")
    for i, (images, targets) in enumerate(test_dataset):
        preds = model.predict(images, verbose=0)
        seg_true.append(targets["segmentation"].numpy())
        seg_prob.append(preds["segmentation"])
        cls_true.append(targets["classification"].numpy())
        cls_prob.append(preds["classification"])
        if i == 0:
            sample_images = images.numpy()
        if (i + 1) % 20 == 0:
            print(f"  Processed {(i+1) * BATCH_SIZE} samples...")

    seg_true = np.concatenate(seg_true, axis=0)
    seg_prob = np.concatenate(seg_prob, axis=0)
    cls_true = np.concatenate(cls_true, axis=0).flatten()
    cls_prob = np.concatenate(cls_prob, axis=0).flatten()
    print(f"  Total: {len(cls_true)} test samples")
    return seg_true, seg_prob, cls_true, cls_prob, sample_images


def plot_confusion_matrix(y_true, y_pred, title, labels, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.5, linecolor='gray')
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=9, color='gray')
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_roc_curve(y_true, y_prob, title, filename):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#2196F3', lw=2.5, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    ax.scatter(fpr[best_idx], tpr[best_idx], c='red', s=100, zorder=5,
               label=f'Best threshold = {thresholds[best_idx]:.3f}')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_precision_recall(y_true, y_prob, title, filename):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='#4CAF50', lw=2.5, label=f'PR Curve (AP = {ap:.4f})')
    ax.fill_between(recall, precision, alpha=0.1, color='#4CAF50')
    ax.axhline(y=np.mean(y_true), color='gray', linestyle='--', alpha=0.5,
               label=f'Baseline (prevalence = {np.mean(y_true):.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.02]); ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_threshold_analysis(y_true, y_prob, title, filename):
    thresholds = np.arange(0.05, 0.96, 0.05)
    f1s, precs, recs, accs = [], [], [], []
    for t in thresholds:
        yp = (y_prob >= t).astype(np.float32)
        f1s.append(f1_score(y_true, yp, zero_division=0))
        precs.append(precision_score(y_true, yp, zero_division=0))
        recs.append(recall_score(y_true, yp, zero_division=0))
        accs.append(accuracy_score(y_true, yp))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, f1s, 'o-', color='#FF5722', lw=2, label='F1 Score', markersize=4)
    ax.plot(thresholds, precs, 's-', color='#2196F3', lw=2, label='Precision', markersize=4)
    ax.plot(thresholds, recs, '^-', color='#4CAF50', lw=2, label='Recall', markersize=4)
    ax.plot(thresholds, accs, 'D-', color='#9C27B0', lw=2, label='Accuracy', markersize=4)
    best_t = thresholds[np.argmax(f1s)]
    ax.axvline(x=best_t, color='red', linestyle='--', alpha=0.7, label=f'Best F1 @ t={best_t:.2f}')
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_prediction_samples(images, seg_true, seg_prob, cls_true, cls_prob, filename):
    n = min(8, len(images))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for j, t in enumerate(['Input Image', 'Ground Truth Mask', 'Predicted Mask', 'Overlay']):
        axes[0, j].set_title(t, fontsize=12, fontweight='bold')
    for i in range(n):
        img = images[i]
        gt = seg_true[i, :, :, 0]
        pred = (seg_prob[i, :, :, 0] >= THRESHOLD).astype(np.float32)
        prob = seg_prob[i, :, :, 0]
        cls_label = "TAMPERED" if cls_true[i] >= 0.5 else "AUTHENTIC"
        cls_pred_label = f"{'TAMPERED' if cls_prob[i] >= 0.5 else 'AUTHENTIC'} ({cls_prob[i]:.3f})"
        axes[i, 0].imshow(np.clip(img, 0, 1))
        axes[i, 0].set_ylabel(f'#{i+1}\nTrue: {cls_label}', fontsize=9, fontweight='bold')
        axes[i, 1].imshow(gt, cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].imshow(pred, cmap='Reds', vmin=0, vmax=1)
        axes[i, 2].set_xlabel(f'Pred: {cls_pred_label}', fontsize=8)
        overlay = np.clip(img.copy(), 0, 1)
        mask_rgb = np.zeros_like(overlay)
        mask_rgb[:, :, 0] = prob
        overlay = 0.6 * overlay + 0.4 * mask_rgb
        axes[i, 3].imshow(np.clip(overlay, 0, 1))
        for j in range(4):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    plt.suptitle('Sample Predictions (Red = Tampered Region)', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_error_distribution(cls_true, cls_prob, filename):
    cls_pred = (cls_prob >= THRESHOLD).astype(int)
    correct = cls_pred == cls_true.astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    auth_probs = cls_prob[cls_true < 0.5]
    tamp_probs = cls_prob[cls_true >= 0.5]
    axes[0].hist(auth_probs, bins=50, alpha=0.7, color='#4CAF50', label='Authentic', density=True)
    axes[0].hist(tamp_probs, bins=50, alpha=0.7, color='#F44336', label='Tampered', density=True)
    axes[0].axvline(x=THRESHOLD, color='black', linestyle='--', lw=2, label=f'Threshold={THRESHOLD}')
    axes[0].set_xlabel('Predicted Probability (Tampered)', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Prediction Distribution by True Class', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)
    correct_probs = cls_prob[correct]
    wrong_probs = cls_prob[~correct]
    axes[1].hist(correct_probs, bins=50, alpha=0.7, color='#2196F3', label=f'Correct (n={len(correct_probs)})', density=True)
    if len(wrong_probs) > 0:
        axes[1].hist(wrong_probs, bins=50, alpha=0.7, color='#FF9800', label=f'Wrong (n={len(wrong_probs)})', density=True)
    axes[1].set_xlabel('Predicted Probability', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Confidence: Correct vs Incorrect', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_segmentation_quality(seg_true, seg_prob, filename):
    n = seg_true.shape[0]
    ious, dices = [], []
    for i in range(n):
        gt = seg_true[i].flatten()
        pred = (seg_prob[i].flatten() >= THRESHOLD).astype(np.float32)
        if gt.sum() == 0 and pred.sum() == 0:
            continue
        intersection = np.sum(gt * pred)
        union = np.sum(gt) + np.sum(pred) - intersection
        ious.append(intersection / (union + 1e-8))
        dices.append(2 * intersection / (np.sum(gt) + np.sum(pred) + 1e-8))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(ious, bins=30, color='#3F51B5', alpha=0.8, edgecolor='white')
    axes[0].axvline(np.mean(ious), color='red', lw=2, linestyle='--', label=f'Mean IoU = {np.mean(ious):.3f}')
    axes[0].set_xlabel('IoU per Image', fontsize=11); axes[0].set_ylabel('Count', fontsize=11)
    axes[0].set_title('Per-Image IoU Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)
    axes[1].hist(dices, bins=30, color='#009688', alpha=0.8, edgecolor='white')
    axes[1].axvline(np.mean(dices), color='red', lw=2, linestyle='--', label=f'Mean Dice = {np.mean(dices):.3f}')
    axes[1].set_xlabel('Dice Score per Image', fontsize=11); axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Per-Image Dice Distribution', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_metrics_summary(seg_m, cls_m, filename):
    names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'MCC']
    seg_v = [seg_m[k] for k in names]
    cls_v = [cls_m[k] for k in names]
    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, seg_v, w, label='Segmentation', color='#2196F3', alpha=0.85, edgecolor='white')
    b2 = ax.bar(x + w/2, cls_v, w, label='Classification', color='#FF5722', alpha=0.85, edgecolor='white')
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance: Segmentation vs Classification', fontsize=14, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=11); ax.set_ylim(0, 1.15); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def generate_report(seg_metrics, cls_metrics, cls_true, cls_prob):
    path = os.path.join(REPORT_DIR, "evaluation_report.txt")
    lines = []
    lines.append("=" * 70)
    lines.append("  TAMPERING DETECTION MODEL — COMPREHENSIVE EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append(f"  Image Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    lines.append(f"  Test Samples: {len(cls_true)}")
    lines.append(f"  Authentic: {int(np.sum(cls_true < 0.5))}")
    lines.append(f"  Tampered:  {int(np.sum(cls_true >= 0.5))}")
    lines.append(f"  Threshold: {THRESHOLD}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  CLASSIFICATION METRICS (Image-level)")
    lines.append("-" * 70)
    for k, v in cls_metrics.items():
        lines.append(f"    {k:30s}: {v:.4f}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  SEGMENTATION METRICS (Pixel-level)")
    lines.append("-" * 70)
    for k, v in seg_metrics.items():
        lines.append(f"    {k:30s}: {v:.4f}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  SKLEARN CLASSIFICATION REPORT")
    lines.append("-" * 70)
    cls_pred = (cls_prob >= THRESHOLD).astype(int)
    lines.append(classification_report(cls_true.astype(int), cls_pred,
                                       target_names=['Authentic', 'Tampered']))
    lines.append("=" * 70)
    report = "\n".join(lines)
    with open(path, 'w') as f:
        f.write(report)
    print(f"\n{report}")
    print(f"  Saved evaluation_report.txt")


def main():
    model, test_dataset = load_model_and_data()
    seg_true, seg_prob, cls_true, cls_prob, sample_images = collect_predictions(model, test_dataset)

    seg_pred_flat = (seg_prob.flatten() >= THRESHOLD).astype(np.float32)
    seg_true_flat = seg_true.flatten()
    cls_pred = (cls_prob >= THRESHOLD).astype(np.float32)

    print("\n" + "=" * 50)
    print("  Computing metrics...")
    print("=" * 50)

    seg_metrics = {
        'Accuracy': accuracy_score(seg_true_flat, seg_pred_flat),
        'Precision': precision_score(seg_true_flat, seg_pred_flat, zero_division=0),
        'Recall': recall_score(seg_true_flat, seg_pred_flat, zero_division=0),
        'F1': f1_score(seg_true_flat, seg_pred_flat, zero_division=0),
        'AUC': roc_auc_score(seg_true_flat, seg_prob.flatten()),
        'MCC': matthews_corrcoef(seg_true_flat, seg_pred_flat),
    }
    cls_metrics = {
        'Accuracy': accuracy_score(cls_true, cls_pred),
        'Precision': precision_score(cls_true, cls_pred, zero_division=0),
        'Recall': recall_score(cls_true, cls_pred, zero_division=0),
        'F1': f1_score(cls_true, cls_pred, zero_division=0),
        'AUC': roc_auc_score(cls_true, cls_prob),
        'MCC': matthews_corrcoef(cls_true.astype(int), cls_pred.astype(int)),
    }

    print("\n" + "=" * 50)
    print("  Generating 12 plots + report...")
    print("=" * 50)

    plot_confusion_matrix(cls_true, cls_pred, 'Classification Confusion Matrix',
                          ['Authentic', 'Tampered'], 'cls_confusion_matrix.png')
    plot_confusion_matrix(seg_true_flat, seg_pred_flat, 'Segmentation Confusion Matrix (Pixel-level)',
                          ['Non-tampered', 'Tampered'], 'seg_confusion_matrix.png')

    plot_roc_curve(cls_true, cls_prob, 'Classification ROC Curve', 'cls_roc_curve.png')
    plot_roc_curve(seg_true_flat, seg_prob.flatten(), 'Segmentation ROC Curve', 'seg_roc_curve.png')

    plot_precision_recall(cls_true, cls_prob, 'Classification Precision-Recall', 'cls_precision_recall.png')
    plot_precision_recall(seg_true_flat, seg_prob.flatten(), 'Segmentation Precision-Recall', 'seg_precision_recall.png')

    plot_threshold_analysis(cls_true, cls_prob, 'Classification Threshold Analysis', 'cls_threshold_analysis.png')
    # Sample seg pixels for speed
    n_px = len(seg_true_flat)
    if n_px > 2_000_000:
        idx = np.random.choice(n_px, 2_000_000, replace=False)
        plot_threshold_analysis(seg_true_flat[idx], seg_prob.flatten()[idx],
                                'Segmentation Threshold Analysis', 'seg_threshold_analysis.png')
    else:
        plot_threshold_analysis(seg_true_flat, seg_prob.flatten(),
                                'Segmentation Threshold Analysis', 'seg_threshold_analysis.png')

    plot_prediction_samples(sample_images, seg_true[:8], seg_prob[:8],
                            cls_true[:8], cls_prob[:8], 'prediction_samples.png')
    plot_error_distribution(cls_true, cls_prob, 'error_distribution.png')
    plot_segmentation_quality(seg_true, seg_prob, 'segmentation_quality.png')
    plot_metrics_summary(seg_metrics, cls_metrics, 'metrics_summary.png')

    generate_report(seg_metrics, cls_metrics, cls_true, cls_prob)

    print("\n" + "=" * 50)
    print(f"  DONE! 13 files saved to: {REPORT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
