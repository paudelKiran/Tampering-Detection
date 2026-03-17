import os
import numpy as np
import tensorflow as tf

# Enable mixed precision for ~2x speedup on Apple M4 GPU
tf.keras.mixed_precision.set_global_policy('mixed_float16')

from model import build_model  # type: ignore
from dataloader import get_datasets
from losses import dice_coefficient
from evaluation import TamperingEvaluation


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "manifests")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
THRESHOLD = 0.5


def get_callbacks():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(PROJECT_ROOT, "logs"),
            histogram_freq=0,
        ),
    ]
    return callbacks


def train_model(model, train_dataset, val_dataset, steps_per_epoch=None):
    """Train the model. build_model() already compiles it with per-output losses."""

    model.summary()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        callbacks=get_callbacks(),
    )
    return history


def evaluate_model(model, test_dataset):
    """Evaluate segmentation and classification on the test set."""

    # Collect segmentation ground truths & predictions
    seg_true_list = []
    seg_prob_list = []
    cls_true_list = []
    cls_prob_list = []

    for images, targets in test_dataset:
        preds = model.predict(images, verbose=0)

        seg_true_list.append(targets["segmentation"].numpy())
        seg_prob_list.append(preds["segmentation"])

        cls_true_list.append(targets["classification"].numpy())
        cls_prob_list.append(preds["classification"])

    # --- Segmentation evaluation ---
    seg_true = np.concatenate(seg_true_list, axis=0).flatten()
    seg_prob = np.concatenate(seg_prob_list, axis=0).flatten()
    seg_pred = (seg_prob >= THRESHOLD).astype(np.float32)

    print("\n=== Segmentation Evaluation ===")
    seg_eval = TamperingEvaluation(seg_true, seg_pred, seg_prob)
    seg_eval.print_all_metrics()
    seg_eval.plot_confusion_matrix()
    seg_eval.plot_roc()

    # --- Classification evaluation ---
    cls_true = np.concatenate(cls_true_list, axis=0).flatten()
    cls_prob = np.concatenate(cls_prob_list, axis=0).flatten()
    cls_pred = (cls_prob >= THRESHOLD).astype(np.float32)

    print("\n=== Classification Evaluation ===")
    cls_eval = TamperingEvaluation(cls_true, cls_pred, cls_prob)
    cls_eval.print_all_metrics()
    cls_eval.plot_confusion_matrix()
    cls_eval.plot_roc()


def main():
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(
        manifest_dir=MANIFEST_DIR,
        batch_size=BATCH_SIZE,
        project_root=PROJECT_ROOT,
    )

    # Subsample training to ~20k samples to fit within 3-hour time budget.
    # 20000 / 32 = 625 steps/epoch × 3 epochs ≈ 2.1 hrs + val ≈ 2.5-3 hrs.
    MAX_TRAIN_STEPS = 625
    print(f"Subsampling training to {MAX_TRAIN_STEPS} steps/epoch ({MAX_TRAIN_STEPS * BATCH_SIZE} samples).")
    print(f"Training for max {EPOCHS} epochs.")

    print("Building model...")
    model = build_model()

    print("Training...")
    train_model(model, train_dataset, val_dataset, steps_per_epoch=MAX_TRAIN_STEPS)

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.keras")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    print("Evaluating on test set...")
    evaluate_model(model, test_dataset)


if __name__ == "__main__":
    main()
