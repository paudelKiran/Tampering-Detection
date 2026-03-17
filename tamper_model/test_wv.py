"""
WV Dataset Testing Script
==========================
Tests the trained tampering detection model on the separate WV dataset.
Generates per-fraud-type analysis with graphs and a comprehensive report.

WV Dataset structure:
  - positive/          → authentic documents
  - fraud1_copy_and_move/
  - fraud2_face_morphing/
  - fraud3_face_replacement/
  - fraud4_combined/
  - fraud5_inpaint_and_rewrite/
  - fraud6_crop_and_replace/

All results saved to evaluation_report/wv_test/
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from model import build_model, FrequencyPreprocessLayer
from dataloader import IMG_SIZE
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
WV_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "WV")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
REPORT_DIR = os.path.join(PROJECT_ROOT, "evaluation_report", "wv_test")
os.makedirs(REPORT_DIR, exist_ok=True)

BATCH_SIZE = 32

CATEGORIES = {
    'positive':                 {'label': 0, 'display': 'Authentic'},
    'fraud1_copy_and_move':     {'label': 1, 'display': 'Copy & Move'},
    'fraud2_face_morphing':     {'label': 1, 'display': 'Face Morphing'},
    'fraud3_face_replacement':  {'label': 1, 'display': 'Face Replacement'},
    'fraud4_combined':          {'label': 1, 'display': 'Combined'},
    'fraud5_inpaint_and_rewrite': {'label': 1, 'display': 'Inpaint & Rewrite'},
    'fraud6_crop_and_replace':  {'label': 1, 'display': 'Crop & Replace'},
}


def load_model():
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
    return model


def load_and_preprocess(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def collect_wv_images():
    """Collect all image paths and labels from the WV dataset."""
    all_paths = []
    all_labels = []
    all_categories = []

    for cat_name, info in CATEGORIES.items():
        cat_dir = os.path.join(WV_DIR, cat_name)
        if not os.path.isdir(cat_dir):
            print(f"  WARNING: {cat_dir} not found, skipping")
            continue
        images = sorted(glob.glob(os.path.join(cat_dir, "*.png")))
        images += sorted(glob.glob(os.path.join(cat_dir, "*.jpg")))
        images += sorted(glob.glob(os.path.join(cat_dir, "*.jpeg")))
        print(f"  {info['display']:25s}: {len(images)} images")
        all_paths.extend(images)
        all_labels.extend([info['label']] * len(images))
        all_categories.extend([cat_name] * len(images))

    return all_paths, all_labels, all_categories


def run_inference(model, image_paths, batch_size=32):
    """Run model inference on all images, return classification probabilities only."""
    cls_probs = []

    n = len(image_paths)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_paths = image_paths[start:end]

        images = tf.stack([load_and_preprocess(p) for p in batch_paths])
        preds = model.predict(images, verbose=0)

        cls_probs.extend(preds['classification'].flatten().tolist())

        batch_num = start // batch_size + 1
        if batch_num % 25 == 0:
            print(f"  Processed {end}/{n} images ({end*100//n}%)...")

    print(f"  Done: {n} images processed")
    return np.array(cls_probs)


# ═══════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════

def plot_overall_confusion_matrix(y_true, y_pred, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues',
                xticklabels=['Authentic', 'Tampered'],
                yticklabels=['Authentic', 'Tampered'],
                ax=ax, linewidths=0.5)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.7, f'({cm[i,j]/total*100:.1f}%)',
                    ha='center', va='center', fontsize=9, color='gray')
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('WV Dataset — Classification Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_roc_curve(y_true, y_prob, filename):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#2196F3', lw=2.5, label=f'ROC (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    j = np.argmax(tpr - fpr)
    ax.scatter(fpr[j], tpr[j], c='red', s=100, zorder=5,
               label=f'Best @ t={thresholds[j]:.3f}')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('WV Dataset — ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_precision_recall(y_true, y_prob, filename):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color='#4CAF50', lw=2.5, label=f'PR (AP = {ap:.4f})')
    ax.fill_between(rec, prec, alpha=0.1, color='#4CAF50')
    ax.set_xlabel('Recall', fontsize=12); ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('WV Dataset — Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_per_fraud_accuracy(categories, cls_true, cls_prob, threshold, filename):
    """Bar chart: accuracy and detection rate per fraud type."""
    cat_names = list(CATEGORIES.keys())
    display_names = [CATEGORIES[c]['display'] for c in cat_names]
    accuracies = []
    detection_rates = []
    counts = []

    cats_arr = np.array(categories)
    cls_pred = (cls_prob >= threshold).astype(int)

    for cat in cat_names:
        mask = cats_arr == cat
        if mask.sum() == 0:
            accuracies.append(0); detection_rates.append(0); counts.append(0)
            continue
        y_t = np.array(cls_true)[mask]
        y_p = cls_pred[mask]
        acc = accuracy_score(y_t, y_p)
        if CATEGORIES[cat]['label'] == 1:
            dr = recall_score(y_t, y_p, zero_division=0)
        else:
            dr = (y_p == 0).mean()  # correct rejection rate
        accuracies.append(acc)
        detection_rates.append(dr)
        counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cat_names))
    w = 0.35
    bars1 = ax.bar(x - w/2, accuracies, w, label='Accuracy', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + w/2, detection_rates, w, label='Detection Rate', color='#FF5722', alpha=0.85)

    for bar, cnt in zip(bars1, counts):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per Fraud Type: Accuracy & Detection Rate', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)

    # Add sample counts
    for i, cnt in enumerate(counts):
        ax.text(i, -0.08, f'n={cnt}', ha='center', fontsize=8, color='gray',
                transform=ax.get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_confidence_by_fraud_type(categories, cls_prob, filename):
    """Box plot of prediction confidence per fraud type."""
    cat_names = list(CATEGORIES.keys())
    display_names = [CATEGORIES[c]['display'] for c in cat_names]
    cats_arr = np.array(categories)

    data = []
    labels = []
    for cat, disp in zip(cat_names, display_names):
        mask = cats_arr == cat
        probs = cls_prob[mask]
        data.append(probs)
        labels.append(disp)

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False)

    colors = ['#4CAF50'] + ['#F44336', '#FF5722', '#FF9800', '#FFC107', '#E91E63', '#9C27B0']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0.5, color='black', linestyle='--', lw=1.5, alpha=0.5, label='Threshold=0.5')
    ax.set_ylabel('Predicted Probability (Tampered)', fontsize=12)
    ax.set_title('Prediction Confidence by Fraud Type', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_error_distribution(cls_true, cls_prob, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    auth = cls_prob[np.array(cls_true) == 0]
    tamp = cls_prob[np.array(cls_true) == 1]
    axes[0].hist(auth, bins=50, alpha=0.7, color='#4CAF50', label='Authentic', density=True)
    axes[0].hist(tamp, bins=50, alpha=0.7, color='#F44336', label='Tampered', density=True)
    axes[0].axvline(x=0.5, color='black', linestyle='--', lw=2, label='Threshold')
    axes[0].set_xlabel('Pred. Probability', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Distribution by True Class', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.3)

    cls_pred = (cls_prob >= 0.5).astype(int)
    correct = cls_pred == np.array(cls_true)
    axes[1].hist(cls_prob[correct], bins=50, alpha=0.7, color='#2196F3',
                 label=f'Correct (n={correct.sum()})', density=True)
    if (~correct).sum() > 0:
        axes[1].hist(cls_prob[~correct], bins=50, alpha=0.7, color='#FF9800',
                     label=f'Wrong (n={(~correct).sum()})', density=True)
    axes[1].set_xlabel('Pred. Probability', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('Confidence: Correct vs Wrong', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_sample_predictions(images, seg_probs, cls_probs, categories, filename):
    n = min(8, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    for j, t in enumerate(['Input Image', 'Tamper Heatmap', 'Overlay']):
        axes[0, j].set_title(t, fontsize=12, fontweight='bold')
    for i in range(n):
        img = images[i]
        prob_map = seg_probs[i][:, :, 0]
        pred_label = 'TAMPERED' if cls_probs[i] >= 0.5 else 'AUTHENTIC'
        cat_disp = CATEGORIES.get(categories[i], {}).get('display', categories[i])

        axes[i, 0].imshow(np.clip(img, 0, 1))
        axes[i, 0].set_ylabel(f'#{i+1}\n{cat_disp}', fontsize=9, fontweight='bold')
        axes[i, 1].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
        axes[i, 1].set_xlabel(f'{pred_label} ({cls_probs[i]:.3f})', fontsize=8)
        overlay = np.clip(img.copy(), 0, 1)
        mask_rgb = np.zeros_like(overlay)
        mask_rgb[:, :, 0] = prob_map
        overlay = 0.6 * overlay + 0.4 * mask_rgb
        axes[i, 2].imshow(np.clip(overlay, 0, 1))
        for j in range(3):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    plt.suptitle('WV Dataset — Sample Predictions', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def generate_report(cls_true, cls_prob, categories, threshold=0.5):
    cls_pred = (cls_prob >= threshold).astype(int)
    cats_arr = np.array(categories)

    lines = []
    lines.append("=" * 70)
    lines.append("  WV DATASET — COMPREHENSIVE TEST REPORT")
    lines.append("=" * 70)
    lines.append(f"  Model: best_model.keras")
    lines.append(f"  Image Size: {IMG_SIZE[0]}x{IMG_SIZE[1]}")
    lines.append(f"  Total Samples: {len(cls_true)}")
    lines.append(f"  Authentic: {int(np.sum(np.array(cls_true)==0))}")
    lines.append(f"  Tampered:  {int(np.sum(np.array(cls_true)==1))}")
    lines.append(f"  Threshold: {threshold}")
    lines.append("")

    lines.append("-" * 70)
    lines.append("  OVERALL CLASSIFICATION METRICS")
    lines.append("-" * 70)
    metrics = {
        'Accuracy': accuracy_score(cls_true, cls_pred),
        'Precision': precision_score(cls_true, cls_pred, zero_division=0),
        'Recall': recall_score(cls_true, cls_pred, zero_division=0),
        'F1 Score': f1_score(cls_true, cls_pred, zero_division=0),
        'AUC-ROC': roc_auc_score(cls_true, cls_prob),
        'MCC': matthews_corrcoef(np.array(cls_true), cls_pred),
    }
    for k, v in metrics.items():
        lines.append(f"    {k:30s}: {v:.4f}")

    lines.append("")
    lines.append("-" * 70)
    lines.append("  SKLEARN CLASSIFICATION REPORT")
    lines.append("-" * 70)
    lines.append(classification_report(cls_true, cls_pred,
                                       target_names=['Authentic', 'Tampered']))

    lines.append("-" * 70)
    lines.append("  PER FRAUD TYPE BREAKDOWN")
    lines.append("-" * 70)

    for cat_name, info in CATEGORIES.items():
        mask = cats_arr == cat_name
        if mask.sum() == 0:
            continue
        y_t = np.array(cls_true)[mask]
        y_p = cls_pred[mask]
        y_prob = cls_prob[mask]
        acc = accuracy_score(y_t, y_p)
        if info['label'] == 1:
            det = recall_score(y_t, y_p, zero_division=0)
            fp = (y_p == 0).sum()
            lines.append(f"\n  {info['display']} ({mask.sum()} samples)")
            lines.append(f"    Accuracy:       {acc:.4f}")
            lines.append(f"    Detection Rate: {det:.4f}")
            lines.append(f"    Missed (FN):    {fp}")
            lines.append(f"    Mean Confidence:{y_prob.mean():.4f}")
        else:
            rej = (y_p == 0).mean()
            fa = (y_p == 1).sum()
            lines.append(f"\n  {info['display']} ({mask.sum()} samples)")
            lines.append(f"    Accuracy:              {acc:.4f}")
            lines.append(f"    Correct Rejection Rate:{rej:.4f}")
            lines.append(f"    False Alarms:          {fa}")
            lines.append(f"    Mean Confidence:       {y_prob.mean():.4f}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("  GENERATED FILES")
    lines.append("=" * 70)
    generated = [
        "wv_confusion_matrix.png", "wv_roc_curve.png",
        "wv_precision_recall.png", "wv_per_fraud_accuracy.png",
        "wv_confidence_boxplot.png", "wv_error_distribution.png",
        "wv_sample_predictions.png", "wv_test_report.txt",
    ]
    for i, f in enumerate(generated, 1):
        lines.append(f"  {i}. {f}")
    lines.append("=" * 70)

    report = "\n".join(lines)
    path = os.path.join(REPORT_DIR, "wv_test_report.txt")
    with open(path, 'w') as f:
        f.write(report)
    print(f"\n{report}")
    print(f"  Saved wv_test_report.txt")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    model = load_model()

    print("\nCollecting WV dataset images...")
    image_paths, labels, categories = collect_wv_images()

    print(f"\nRunning inference on {len(image_paths)} images...")
    cls_prob = run_inference(model, image_paths, BATCH_SIZE)

    cls_true = np.array(labels)
    cls_pred = (cls_prob >= 0.5).astype(int)

    print("\n" + "=" * 50)
    print("  Generating plots & report...")
    print("=" * 50)

    plot_overall_confusion_matrix(cls_true, cls_pred, 'wv_confusion_matrix.png')
    plot_roc_curve(cls_true, cls_prob, 'wv_roc_curve.png')
    plot_precision_recall(cls_true, cls_prob, 'wv_precision_recall.png')
    plot_per_fraud_accuracy(categories, cls_true, cls_prob, 0.5, 'wv_per_fraud_accuracy.png')
    plot_confidence_by_fraud_type(categories, cls_prob, 'wv_confidence_boxplot.png')
    plot_error_distribution(cls_true, cls_prob, 'wv_error_distribution.png')

    # Sample predictions — pick one from each category
    sample_indices = []
    cats_arr = np.array(categories)
    for cat in CATEGORIES:
        idx = np.where(cats_arr == cat)[0]
        if len(idx) > 0:
            sample_indices.append(idx[0])

    sample_imgs = []
    sample_segs = []
    sample_cls = []
    sample_cats = []
    for idx in sample_indices[:8]:
        img = load_and_preprocess(image_paths[idx]).numpy()
        preds = model.predict(tf.expand_dims(img, 0), verbose=0)
        sample_imgs.append(img)
        sample_segs.append(preds['segmentation'][0])
        sample_cls.append(cls_prob[idx])
        sample_cats.append(categories[idx])

    plot_sample_predictions(sample_imgs, sample_segs, sample_cls, sample_cats,
                            'wv_sample_predictions.png')

    generate_report(cls_true, cls_prob, categories)

    print("\n" + "=" * 50)
    print(f"  ALL DONE! 8 files saved to: {REPORT_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
