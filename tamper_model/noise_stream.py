import tensorflow as tf

try:
    from tamper_model.bayerconv_layer import BayarConv
except ImportError:
    from bayerconv_layer import BayarConv


# ---------------------------------------------------------------------------
# ResNet-18 residual block
# ---------------------------------------------------------------------------

def residual_block(x, filters, stride=1):
    """Single ResNet-18 residual block (two 3×3 convolutions + skip).

    When stride > 1 or the channel count changes, a 1×1 projection shortcut
    is used to match dimensions before the addition.
    """

    shortcut = x

    # First conv — may downsample spatially when stride = 2
    x = tf.keras.layers.Conv2D(
        filters, 3, strides=stride, padding='same', use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Second conv — always stride 1
    x = tf.keras.layers.Conv2D(
        filters, 3, strides=1, padding='same', use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Projection shortcut when spatial dims or channels change
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(
            filters, 1, strides=stride, padding='same', use_bias=False,
        )(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    # Skip connection followed by activation
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)

    return x


# ---------------------------------------------------------------------------
# ResNet-18 backbone — four stages producing multi-scale feature maps
# ---------------------------------------------------------------------------

def noise_resnet18_backbone(x):
    """ResNet-18 backbone that returns feature maps at four scales.

    Each stage contains two residual blocks.  The first block of every stage
    uses stride 2 to halve the spatial resolution.

    Assuming input at 1/2 resolution (after the stem):
        Stage 1 → f1 at 1/4   (64 channels)
        Stage 2 → f2 at 1/8   (128 channels)
        Stage 3 → f3 at 1/16  (256 channels)
        Stage 4 → f4 at 1/32  (512 channels)
    """

    # Stage 1 — 64 filters, output at 1/4
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64, stride=1)
    f1 = x

    # Stage 2 — 128 filters, output at 1/8
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128, stride=1)
    f2 = x

    # Stage 3 — 256 filters, output at 1/16
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256, stride=1)
    f3 = x

    # Stage 4 — 512 filters, output at 1/32
    x = residual_block(x, 512, stride=2)
    x = residual_block(x, 512, stride=1)
    f4 = x

    return f1, f2, f3, f4


# ---------------------------------------------------------------------------
# Full noise stream: BayarConv → stem → ResNet-18
# ---------------------------------------------------------------------------

def noise_stream(input_image):
    """Complete noise feature-extraction pipeline.

    input_image
        ↓  BayarConv          (suppress content, amplify noise)
        ↓  Conv + BN + ReLU   (initial feature extraction, full res)
        ↓  Stride-2 Conv + BN + ReLU  (downsample to 1/2, replaces pooling)
        ↓  ResNet-18 backbone
        → f1 (1/4), f2 (1/8), f3 (1/16), f4 (1/32)
    """

    # Suppress image content and amplify sensor noise residuals
    x = BayarConv(filters=3)(input_image)

    # Initial feature extraction — keep full resolution
    x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Stride-2 downsample to 1/2 (no pooling — preserves noise information)
    x = tf.keras.layers.Conv2D(
        64, 3, strides=2, padding='same', use_bias=False,
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # Multi-scale feature extraction via ResNet-18
    f1, f2, f3, f4 = noise_resnet18_backbone(x)

    return f1, f2, f3, f4


# ---------------------------------------------------------------------------
# Model wrapper (keeps the interface used by model.py)
# ---------------------------------------------------------------------------

def noise_backbone(input_shape):
    """Build a Keras Model wrapping the full noise stream.

    Returns a Model that accepts an image tensor of *input_shape* and
    produces four multi-scale feature maps [f1, f2, f3, f4].
    """

    inputs = tf.keras.Input(shape=input_shape)
    f1, f2, f3, f4 = noise_stream(inputs)
    return tf.keras.Model(inputs, [f1, f2, f3, f4], name='noise_stream')




