import tensorflow as tf


def classification_head(fused_features):
    """
    Image-level classification head for binary tampering detection.

    Takes the fused feature map from the Multistream Feature Fusion module
    and produces a single scalar authenticity score per image.

    Args:
        fused_features: Tensor of shape (batch_size, H, W, C) — the spatially
                        fused output from texture, frequency, and noise streams.

    Returns:
        Tensor of shape (batch_size, 1) with values in [0, 1].
        0 → authentic image, 1 → tampered image.
    """

    # Collapse spatial dimensions into a single feature vector per image
    x = tf.keras.layers.GlobalAveragePooling2D(
        name='cls_global_avg_pool',
    )(fused_features)

    # Project into a compact representation with non-linear activation
    x = tf.keras.layers.Dense(
        128, activation='relu', name='cls_dense',
    )(x)

    # Regularise to reduce overfitting on small forensic datasets
    x = tf.keras.layers.Dropout(0.5, name='cls_dropout')(x)

    # Binary output: probability that the image is tampered
    x = tf.keras.layers.Dense(
        1, activation='sigmoid', name='classification', dtype='float32',
    )(x)

    return x


# ── Example usage ────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Simulate a fused feature map coming from the encoder / fusion module
    inputs = tf.keras.Input(shape=(64, 64, 256), name='fused_features')

    # Build the classification head on top
    output = classification_head(inputs)

    model = tf.keras.Model(inputs=inputs, outputs=output,
                           name='classification_head_demo')
    model.summary()
