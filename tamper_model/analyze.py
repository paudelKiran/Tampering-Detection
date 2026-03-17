"""
Standalone evaluation script.
Loads the best saved checkpoint and runs full metrics on the test split.

Usage:
    cd tamper_model
    /opt/anaconda3/envs/tf-gpu/bin/python analyze.py
"""
import os
import sys
import warnings
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")          # headless – saves to PNG instead of opening windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, matthews_corrcoef,
    roc_curve, precision_recall_curve, average_precision_score,
)

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_DIR  = os.path.join(PROJECT_ROOT, "manifests")
CHECKPOINT    = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.keras")
REPORT_DIR    = os.path.join(PROJECT_ROOT, "evaluation_report")
os.makedirs(REPORT_DIR, exist_ok=True)

BATCH_SIZE  = 4
THRESHOLD   = 0.5

sys.path.insert(0, os.path.dirname(__file__))
from dataloader import get_datasets
from losses import segmentation_loss, classification_loss, dice_coefficient, focal_loss, tversky_loss
from model import build_model

# ── load model ─────────────────────────────────────────────────────────────
print(f"\nLoading checkpoint: {CHECKPOINT}")
# Rebuild architecture then load weights — avoids custom-layer deserialisation issues
model = build_model()
model.load_weights(CHECKPOINT)
print("Model loaded.\n")

# ── load test data ─────────────────────────────────────────────────────────
print("Loading test dataset (full split)...")
_, _, test_ds = get_datasets(
    manifest_dir=MANIFEST_DIR,
    batch_size=BATCH_SIZE,
    project_root=PROJECT_ROOT,
)

# ── run inference ──────────────────────────────────────────────────────────
print("Running inference on test set...")
seg_true_list, seg_prob_list = [], []
cls_true_list, cls_prob_list = [], []

for batch_idx, (images, targets) in enumerate(test_ds):
    preds = model.predict(images, verbose=0)
    seg_true_list.append(targets["segmentation"].numpy())
    seg_prob_list.append(preds["segmentation"])
    cls_true_list.append(targets["classification"].numpy())
    cls_prob_list.append(preds["classification"])
    if (batch_idx + 1) % 50 == 0:
        print(f"  processed {(batch_idx+1)*BATCH_SIZE} images...")

seg_true = np.concatenate(seg_true_list, axis=0).flatten()
seg_prob = np.concatenate(seg_prob_list, axis=0).flatten()
seg_pred = (seg_prob >= THRESHOLD).astype(np.float32)

cls_true = np.concatenate(cls_true_list, axis=0).flatten()
cls_prob = np.concatenate(cls_prob_list, axis=0).flatten()
cls_pred = (cls_prob >= THRESHOLD).astype(np.float32)

print("Inference done.\n")

# ── helpers ────────────────────────────────────────────────────────────────

def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_prob)

def safe_ap(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return average_precision_score(y_true, y_prob)

def print_metrics(name, y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred)
    print(f"{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"\nConfusion Matrix:\n{cm}")

    labels = np.unique(y_true)
    has_pos = 1 in labels

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)
    auc  = safe_auc(y_true, y_prob)
    ap   = safe_ap(y_true, y_prob)

    # IoU & Dice over full pixel arrays
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou   = inter / union if union > 0 else 0.0
    denom = y_true.sum() + y_pred.sum()
    dice  = (2 * inter) / denom if denom > 0 else 0.0

    # pixel-level class counts
    n_neg = int((y_true == 0).sum())
    n_pos = int((y_true == 1).sum())
    total = n_neg + n_pos

    print(f"\nClass distribution:")
    print(f"  Negative (0): {n_neg:>12,}  ({100*n_neg/total:.1f}%)")
    print(f"  Positive (1): {n_pos:>12,}  ({100*n_pos/total:.1f}%)")

    print(f"\nCore metrics:")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Precision     : {prec:.4f}")
    print(f"  Recall        : {rec:.4f}")
    print(f"  F1 Score      : {f1:.4f}")
    print(f"  MCC           : {mcc:.4f}")

    print(f"\nSegmentation metrics:")
    print(f"  IoU (Jaccard) : {iou:.4f}")
    print(f"  Dice Score    : {dice:.4f}")

    print(f"\nRanking metrics:")
    print(f"  ROC-AUC       : {auc:.4f}" if not np.isnan(auc) else "  ROC-AUC       : N/A (single class in y_true)")
    print(f"  Avg Precision : {ap:.4f}"  if not np.isnan(ap)  else "  Avg Precision : N/A")
    print()
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, mcc=mcc,
                iou=iou, dice=dice, auc=auc, ap=ap, cm=cm,
                y_true=y_true, y_pred=y_pred, y_prob=y_prob)

seg_metrics = print_metrics("SEGMENTATION (pixel-level)",
                             seg_true, seg_pred, seg_prob)
cls_metrics = print_metrics("CLASSIFICATION (image-level)",
                             cls_true, cls_pred, cls_prob)

# ── plots ──────────────────────────────────────────────────────────────────

def save_confusion_matrix(cm, title, path, classes=("Authentic","Tampered")):
    fig, ax = plt.subplots(figsize=(6, 5))
    # normalised version for colour, raw counts for text
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax,
                vmin=0, vmax=1)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def save_roc(y_true, y_prob, title, path):
    if np.isnan(safe_auc(y_true, y_prob)):
        print(f"Skipping ROC for {title} (single class in y_true)")
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0,1],[0,1],"--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def save_pr_curve(y_true, y_prob, title, path):
    if np.isnan(safe_ap(y_true, y_prob)):
        print(f"Skipping PR curve for {title}")
        return
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def save_threshold_sweep(y_true, y_prob, title, path):
    if np.isnan(safe_ap(y_true, y_prob)):
        return
    thresholds = np.linspace(0, 1, 101)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        yp = (y_prob >= t).astype(np.float32)
        f1s.append(f1_score(y_true, yp, zero_division=0))
        precs.append(precision_score(y_true, yp, zero_division=0))
        recs.append(recall_score(y_true, yp, zero_division=0))
    best_t = thresholds[np.argmax(f1s)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1s,   label="F1")
    ax.plot(thresholds, precs, label="Precision")
    ax.plot(thresholds, recs,  label="Recall")
    ax.axvline(best_t, color="red", linestyle="--",
               label=f"Best F1 threshold = {best_t:.2f}")
    ax.set_xlabel("Threshold"); ax.set_title(title); ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}  (best threshold = {best_t:.2f})")

print("\nGenerating plots...")

save_confusion_matrix(seg_metrics["cm"],
    "Segmentation Confusion Matrix",
    os.path.join(REPORT_DIR, "seg_confusion_matrix.png"))

save_confusion_matrix(cls_metrics["cm"],
    "Classification Confusion Matrix",
    os.path.join(REPORT_DIR, "cls_confusion_matrix.png"),
    classes=("Authentic", "Tampered"))

save_roc(seg_true, seg_prob,
    "Segmentation ROC Curve",
    os.path.join(REPORT_DIR, "seg_roc.png"))

save_roc(cls_true, cls_prob,
    "Classification ROC Curve",
    os.path.join(REPORT_DIR, "cls_roc.png"))

save_pr_curve(seg_true, seg_prob,
    "Segmentation Precision-Recall Curve",
    os.path.join(REPORT_DIR, "seg_pr_curve.png"))

save_pr_curve(cls_true, cls_prob,
    "Classification Precision-Recall Curve",
    os.path.join(REPORT_DIR, "cls_pr_curve.png"))

save_threshold_sweep(seg_true, seg_prob,
    "Segmentation: F1 / Precision / Recall vs Threshold",
    os.path.join(REPORT_DIR, "seg_threshold_sweep.png"))

save_threshold_sweep(cls_true, cls_prob,
    "Classification: F1 / Precision / Recall vs Threshold",
    os.path.join(REPORT_DIR, "cls_threshold_sweep.png"))

print(f"\nAll reports saved to: {REPORT_DIR}/")
print("Done.")
