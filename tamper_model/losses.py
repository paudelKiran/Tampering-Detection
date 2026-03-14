import tensorflow as tf


# Small constants for numerical stability
EPSILON = 1e-7
SMOOTH = 1.0


# ---------------------------------------------------------------------------
# Dice loss & coefficient  (pixel-level overlap measure)
# ---------------------------------------------------------------------------

def dice_loss(y_true, y_pred):
    """1 − Dice coefficient.  Measures spatial overlap between predicted and
    true tampering masks.  Lower is better."""

    # Clip predictions to avoid log(0) in downstream BCE and div-by-zero here
    y_pred = tf.clip_by_value(y_pred, EPSILON, 1.0 - EPSILON)

    # Flatten spatial dims so the formula works for any (H, W)
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    # SMOOTH prevents 0/0 when both mask and prediction are empty
    return 1.0 - (2.0 * intersection + SMOOTH) / (denominator + SMOOTH)


def dice_coefficient(y_true, y_pred):
    """Dice coefficient metric (complement of dice_loss)."""

    y_pred = tf.clip_by_value(y_pred, EPSILON, 1.0 - EPSILON)
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    return (2.0 * intersection + SMOOTH) / (denominator + SMOOTH)


# ---------------------------------------------------------------------------
# Segmentation loss  (BCE + Dice)
# ---------------------------------------------------------------------------

def segmentation_loss(y_true, y_pred):
    """Pixel-level loss for the tampering mask.

    Combines binary cross-entropy (strong gradient signal per pixel) with
    Dice loss (global overlap signal that handles class imbalance well).

    segmentation_loss = BCE + Dice
    """

    y_pred = tf.clip_by_value(y_pred, EPSILON, 1.0 - EPSILON)
    y_true = tf.cast(y_true, tf.float32)

    # Per-pixel binary cross-entropy, averaged over all pixels
    bce = -(y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    bce = tf.reduce_mean(bce)

    # Dice component for global spatial-overlap awareness
    dice = dice_loss(y_true, y_pred)

    return bce + dice


# ---------------------------------------------------------------------------
# Classification loss  (BCE for image-level label)
# ---------------------------------------------------------------------------

def classification_loss(y_true, y_pred):
    """Image-level binary cross-entropy.

    y_true / y_pred have shape (batch_size, 1).
    0 → authentic, 1 → tampered.
    """

    y_pred = tf.clip_by_value(y_pred, EPSILON, 1.0 - EPSILON)
    y_true = tf.cast(y_true, tf.float32)

    bce = -(y_true * tf.math.log(y_pred)
            + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(bce)


# ---------------------------------------------------------------------------
# Combined multi-task loss
# ---------------------------------------------------------------------------

def combined_loss(mask_true, mask_pred, label_true, label_pred):
    """Total training loss for the two-output model.

    total_loss = segmentation_loss + classification_loss
    """

    seg_loss = segmentation_loss(mask_true, mask_pred)
    cls_loss = classification_loss(label_true, label_pred)
    return seg_loss + cls_loss


# ---------------------------------------------------------------------------
# Helper to retrieve per-output loss dict for model.compile()
# ---------------------------------------------------------------------------

def get_loss_functions():
    """Return a dict suitable for `model.compile(loss=...)`
    when the model outputs {'segmentation': ..., 'classification': ...}."""

    return {
        'segmentation': segmentation_loss,
        'classification': classification_loss,
    }


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # --- build a tiny two-output model for demonstration ---
    inp = tf.keras.Input(shape=(64, 64, 3), name='input_image')

    # Segmentation branch (dummy)
    mask_output = tf.keras.layers.Conv2D(
        1, 1, activation='sigmoid', name='segmentation',
    )(inp)

    # Classification branch (dummy)
    x = tf.keras.layers.GlobalAveragePooling2D()(inp)
    class_output = tf.keras.layers.Dense(
        1, activation='sigmoid', name='classification',
    )(x)

    model = tf.keras.Model(
        inputs=inp,
        outputs={'segmentation': mask_output, 'classification': class_output},
    )

    # Compile using the per-output loss dict
    model.compile(
        optimizer='adam',
        loss=get_loss_functions(),
        metrics={
            'segmentation': [dice_coefficient, 'accuracy'],
            'classification': ['accuracy'],
        },
    )

    model.summary()
    print('\nLoss dict:', get_loss_functions())
