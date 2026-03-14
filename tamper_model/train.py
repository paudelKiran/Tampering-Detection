import os
import numpy as np
import tensorflow as tf

from model import build_model # type: ignore
from dataloader import get_datasets
from losses import combined_loss, dice_coefficient
from evaluation import TamperingEvaluation


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "manifests")


TRAIN_TAKE = 3
VAL_TAKE = 2
TEST_TAKE = 2
EPOCHS = 2
LEARNING_RATE = 1e-4
THRESHOLD = 0.5


def train_model(model, train_dataset, val_dataset):

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=combined_loss,
        metrics=["accuracy", dice_coefficient],
    )

    model.summary()

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
    )
    return history


def evaluate_model(model, test_dataset):

    all_y_true = []
    all_y_prob = []

    for images, masks in test_dataset:
        preds = model.predict(images, verbose=0)
        all_y_true.append(masks.numpy())
        all_y_prob.append(preds)

    y_true = np.concatenate(all_y_true, axis=0).flatten()
    y_prob = np.concatenate(all_y_prob, axis=0).flatten()
    y_pred = (y_prob >= THRESHOLD).astype(np.float32)

    evaluator = TamperingEvaluation(y_true, y_pred, y_prob)
    evaluator.print_all_metrics()
    evaluator.plot_confusion_matrix()
    evaluator.plot_roc()


def main():
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = get_datasets(
        manifest_dir=MANIFEST_DIR, project_root=PROJECT_ROOT
    )


    train_dataset = train_dataset.take(TRAIN_TAKE)
    val_dataset = val_dataset.take(VAL_TAKE)
    test_dataset = test_dataset.take(TEST_TAKE)

    print("Building model...")
    model = build_model()

    print("Training...")
    train_model(model, train_dataset, val_dataset)

    print("Evaluating on test set...")
    evaluate_model(model, test_dataset)


if __name__ == "__main__":
    main()
