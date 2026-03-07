"""
Segmentation Head for Image Tampering Detection

U-Net style decoder that takes fused multi-stream encoder features and
produces a pixel-level binary tampering mask.

Inputs:
    fused_features  – (B, H/8, W/8, C)   from final_fusion()
    encoder_features – [skip1, skip2, skip3]
        skip1 → (B, H/4,  W/4,  C1)      high-resolution skip
        skip2 → (B, H/8,  W/8,  C2)      mid-resolution skip
        skip3 → (B, H/16, W/16, C3)      low-resolution skip

Output:
    tampering_mask – (B, H, W, 1)   values in [0, 1]
        0 = authentic pixel
        1 = tampered pixel
"""

import tensorflow as tf


# ---------------------------------------------------------------------------
# Helper: two-conv block with BatchNorm + ReLU
# ---------------------------------------------------------------------------
def conv_block(x, filters):
    """Apply two successive Conv2D → BatchNorm → ReLU operations.

    Args:
        x:       input tensor.
        filters: number of output filters for both convolutions.

    Returns:
        Tensor after two conv-bn-relu stages.
    """
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x


# ---------------------------------------------------------------------------
# Helper: single decoder stage  (upsample → concat skip → conv_block)
# ---------------------------------------------------------------------------
def decoder_block(x, skip, filters):
    """One U-Net decoder stage.

    1. Bilinear upsample x by 2×.
    2. Concatenate with the corresponding encoder skip connection.
    3. Refine with conv_block (two Conv-BN-ReLU layers).

    Args:
        x:       input tensor from the previous decoder stage.
        skip:    encoder skip-connection tensor at the matching resolution.
        filters: number of filters for the conv_block.

    Returns:
        Decoded feature tensor at 2× the spatial size of x.
    """
    # Upsample spatially by factor 2
    x = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation="bilinear"
    )(x)

    # Concatenate with skip connection along channel axis
    x = tf.keras.layers.Concatenate(axis=-1)([x, skip])

    # Refine with two conv-bn-relu layers
    x = conv_block(x, filters)

    return x


# ---------------------------------------------------------------------------
# Segmentation head (full decoder)
# ---------------------------------------------------------------------------
def segmentation_head(fused_features, encoder_features):
    """U-Net style decoder producing a binary tampering mask.

    Decoder path (spatial sizes assume an original image of H×W):

        fused_features  (H/8,  W/8,  C)
            │
            ├── upsample 2× ──► concat skip3 (H/8 after upsampling H/16 skip
            │   is not needed here; skip3 lives at H/16 so we first go *down*
            │   then back up.  See note below.)
            │
        Stage 1:  process fused + skip2 at  H/8   → decode to H/4
        Stage 2:  process          + skip1 at H/4  → decode to H/2
        Stage 3:  upsample to full H  (no skip)    → decode to H
            │
            └── 1×1 Conv + sigmoid  →  (H, W, 1)

    Because fused_features already sits at H/8 and skip3 is at H/16, we
    first *downsample-merge* with skip3, then walk back up through skip2
    and skip1 to reach full resolution.

    Args:
        fused_features:   tensor of shape (B, H/8, W/8, C).
        encoder_features: list [skip1, skip2, skip3] where
            skip1 → (B, H/4,  W/4,  C1)
            skip2 → (B, H/8,  W/8,  C2)
            skip3 → (B, H/16, W/16, C3)

    Returns:
        tampering_mask: tensor of shape (B, H, W, 1).
    """
    skip1, skip2, skip3 = encoder_features  # unpack

    # ------------------------------------------------------------------
    # Stage 0 – Merge fused features with deepest skip (skip3 at H/16).
    # Downsample fused (H/8) to H/16 via strided conv, concat skip3,
    # then refine.  This gives the decoder a "bottleneck" that sees
    # the lowest-resolution encoder information.
    # ------------------------------------------------------------------
    bottleneck = tf.keras.layers.Conv2D(
        256, 3, strides=2, padding="same", use_bias=False
    )(fused_features)
    bottleneck = tf.keras.layers.BatchNormalization()(bottleneck)
    bottleneck = tf.keras.layers.ReLU()(bottleneck)

    bottleneck = tf.keras.layers.Concatenate(axis=-1)([bottleneck, skip3])
    bottleneck = conv_block(bottleneck, 256)

    # ------------------------------------------------------------------
    # Stage 1 – Upsample H/16 → H/8, concat skip2, refine
    # ------------------------------------------------------------------
    x = decoder_block(bottleneck, skip2, 128)

    # ------------------------------------------------------------------
    # Stage 2 – Upsample H/8 → H/4, concat skip1, refine
    # ------------------------------------------------------------------
    x = decoder_block(x, skip1, 64)

    # ------------------------------------------------------------------
    # Stage 3 – Upsample H/4 → H/2, refine (no skip connection)
    # ------------------------------------------------------------------
    x = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation="bilinear"
    )(x)
    x = conv_block(x, 32)

    # ------------------------------------------------------------------
    # Stage 4 – Upsample H/2 → H, refine (no skip connection)
    # ------------------------------------------------------------------
    x = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation="bilinear"
    )(x)
    x = conv_block(x, 16)

    # ------------------------------------------------------------------
    # Output – 1×1 convolution with sigmoid → binary mask
    # ------------------------------------------------------------------
    tampering_mask = tf.keras.layers.Conv2D(
        1, kernel_size=1, padding="same", activation="sigmoid"
    )(x)

    return tampering_mask


# -----------------------------------------------------------------------
# Quick smoke test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    H, W = 256, 256

    # Simulated inputs matching the described shapes
    fused   = tf.keras.Input(shape=(H // 8, W // 8, 256), name="fused")
    skip1   = tf.keras.Input(shape=(H // 4, W // 4, 64),  name="skip1")
    skip2   = tf.keras.Input(shape=(H // 8, W // 8, 128), name="skip2")
    skip3   = tf.keras.Input(shape=(H // 16, W // 16, 256), name="skip3")

    mask = segmentation_head(fused, [skip1, skip2, skip3])

    model = tf.keras.Model(
        inputs=[fused, skip1, skip2, skip3],
        outputs=mask,
        name="segmentation_head",
    )
    model.summary()

    # Verify output shape with a dummy batch
    import numpy as np

    dummy = [
        np.random.rand(2, H // 8, W // 8, 256).astype("float32"),
        np.random.rand(2, H // 4, W // 4, 64).astype("float32"),
        np.random.rand(2, H // 8, W // 8, 128).astype("float32"),
        np.random.rand(2, H // 16, W // 16, 256).astype("float32"),
    ]
    out = model.predict(dummy, verbose=0)
    print(f"\nOutput shape: {out.shape}  (expected: (2, {H}, {W}, 1))")
    assert out.shape == (2, H, W, 1), "Shape mismatch!"
    print("All checks passed.")
