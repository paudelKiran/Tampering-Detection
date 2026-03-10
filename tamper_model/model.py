"""
Multistream Tampering Detection Model
======================================

Integrates three parallel feature extraction streams (texture, frequency, noise),
performs multistream feature fusion, and outputs a pixel-level tampering
segmentation mask via a U-Net style decoder.

Pipeline:
    Input Image (512x512x3)
    ├─► Texture Stream   → [Dt1, Dt2, Dt3, Dt4]
    ├─► Frequency Stream → [Df1, Df2, Df3, Df4]
    └─► Noise Stream     → [Dn1, Dn2, Dn3, Dn4]
         │
         ▼
    Multistream Feature Fusion → fused (64x64x256)
         │
         ▼
    Segmentation Head (+ texture skip connections)
         │
         ▼
    Tampering Mask (512x512x1)
"""

import tensorflow as tf

try:
    from tamper_model.texture_stream import texture_stream
    from tamper_model.frequenct_Stream import (
        rgb_to_ycbcr,
        block_dct,
        reconstruct_dct_volume,
        simulate_quantization,
        binary_volume_encoding,
        frequency_backbone_hrnet_w18,
    )
    from tamper_model.noise_stream import noise_backbone
    from tamper_model.feature_fusion import (
        fuse_texture_frequency,
        fuse_noise,
        final_fusion,
    )
    from tamper_model.segmentation import segmentation_head
except ImportError:
    from texture_stream import texture_stream
    from frequenct_Stream import (
        rgb_to_ycbcr,
        block_dct,
        reconstruct_dct_volume,
        simulate_quantization,
        binary_volume_encoding,
        frequency_backbone_hrnet_w18,
    )
    from noise_stream import noise_backbone
    from feature_fusion import fuse_texture_frequency, fuse_noise, final_fusion
    from segmentation import segmentation_head


# ---------------------------------------------------------------------------
# Custom Keras layers (wrap raw TF ops for Keras 3 compatibility)
# ---------------------------------------------------------------------------

class FrequencyPreprocessLayer(tf.keras.layers.Layer):
    """DCT-based preprocessing: RGB → YCbCr → block DCT → binary volume."""

    def compute_output_shape(self, input_shape): # type: ignore
        h = input_shape[1]
        w = input_shape[2]
        out_h = None if h is None else h // 8
        out_w = None if w is None else w // 8
        return (input_shape[0], out_h, out_w, 320)

    def call(self, image):
        H = tf.shape(image)[1]
        W = tf.shape(image)[2]
        ycbcr = rgb_to_ycbcr(image)
        Y = ycbcr[..., 0]
        dct_blocks = block_dct(Y)
        dct_volume = reconstruct_dct_volume(dct_blocks, H, W)
        dct_volume = simulate_quantization(dct_volume)
        binary_volume = binary_volume_encoding(dct_volume)
        return binary_volume


# ---------------------------------------------------------------------------
# Multistream feature fusion
# ---------------------------------------------------------------------------

def multistream_feature_fusion(texture_features, frequency_features, noise_features):
    """
    Fuse features from the three parallel streams.

    Steps:
        1. Align frequency feature spatial dimensions to match the texture stream.
        2. Fuse texture + frequency via progressive upsampling.
        3. Fuse noise features via progressive upsampling.
        4. Cross-stream fusion of the two fused representations.
        5. Downsample to the resolution expected by the segmentation head.

    Args:
        texture_features:   [Dt1, Dt2, Dt3, Dt4]  (256, 128, 64, 32 for 512 input)
        frequency_features: List of 4 HRNet outputs from the DCT branch.
        noise_features:     [Dn1, Dn2, Dn3, Dn4]  (512, 256, 128, 64 for 512 input)

    Returns:
        Fused feature tensor at 64x64 spatial resolution (for 512x512 input).
    """
    Dt1, Dt2, Dt3, Dt4 = texture_features
    Dn1, Dn2, Dn3, Dn4 = noise_features

    # --- Align frequency features to texture spatial dimensions ---
    # The DCT-based frequency stream operates on a smaller spatial grid;
    # bilinear resize brings each scale to the matching texture resolution.
    Df2 = tf.keras.layers.Resizing(
        Dt2.shape[1], Dt2.shape[2], interpolation='bilinear',
    )(frequency_features[0])
    Df3 = tf.keras.layers.Resizing(
        Dt3.shape[1], Dt3.shape[2], interpolation='bilinear',
    )(frequency_features[1])
    Df4 = tf.keras.layers.Resizing(
        Dt4.shape[1], Dt4.shape[2], interpolation='bilinear',
    )(frequency_features[2])

    # --- Texture–frequency fusion (output at Dt1 resolution, e.g. 256x256) ---
    Dtf = fuse_texture_frequency(Dt1, Dt2, Dt3, Dt4, Df2, Df3, Df4)

    # --- Noise fusion (output at Dn1 resolution, e.g. 512x512) ---
    Dn_fused = fuse_noise(Dn1, Dn2, Dn3, Dn4)

    # --- Match spatial dimensions for cross-stream fusion ---
    Dn_matched = tf.keras.layers.Resizing(
        Dtf.shape[1], Dtf.shape[2], interpolation='bilinear',
    )(Dn_fused)

    # --- Final cross-stream fusion (Conv2D 256 filters) ---
    fused = final_fusion(Dn_matched, Dtf)

    # --- Downsample from 256x256 → 64x64 for segmentation head ---
    # Two stride-2 convolutions: 256→128→64
    fused = tf.keras.layers.Conv2D(
        256, 3, strides=2, padding='same', use_bias=False,
        name='fusion_down1_conv',
    )(fused)
    fused = tf.keras.layers.BatchNormalization(name='fusion_down1_bn')(fused)
    fused = tf.keras.layers.ReLU(name='fusion_down1_relu')(fused)

    fused = tf.keras.layers.Conv2D(
        256, 3, strides=2, padding='same', use_bias=False,
        name='fusion_down2_conv',
    )(fused)
    fused = tf.keras.layers.BatchNormalization(name='fusion_down2_bn')(fused)
    fused = tf.keras.layers.ReLU(name='fusion_down2_relu')(fused)

    return fused


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(input_shape=(512, 512, 3)):
    """
    Build and compile the multistream tampering-detection model.

    Architecture:
        Input Image
        → Texture Stream   (HRNet-W18)
        → Frequency Stream  (DCT + HRNet-W18)
        → Noise Stream      (BayarConv + CNN)
        → Multistream Feature Fusion
        → Segmentation Head (U-Net decoder with texture skip connections)
        → Tampering Mask (sigmoid, same spatial size as input)

    Returns:
        Compiled ``tf.keras.Model`` with Adam optimiser and binary
        cross-entropy loss.
    """
    inputs = tf.keras.Input(shape=input_shape, name='input_image')

    # ── Stream 1: Texture ──────────────────────────────────────────────
    # Returns [Dt1 (256), Dt2 (128), Dt3 (64), Dt4 (32)]
    texture_features = texture_stream(inputs)

    # ── Stream 2: Frequency ────────────────────────────────────────────
    # DCT preprocessing wrapped in a custom layer (Keras 3 compatible),
    # then the HRNet-W18 backbone extracts multi-scale features.
    freq_preprocessed = FrequencyPreprocessLayer(
        name='frequency_preprocess',
    )(inputs)
    freq_backbone = frequency_backbone_hrnet_w18()
    frequency_features = freq_backbone(freq_preprocessed)

    # ── Stream 3: Noise ────────────────────────────────────────────────
    # Rescaling layer normalises to [0, 1], then BayarConv + CNN backbone.
    noise_normalized = tf.keras.layers.Rescaling(
        1.0 / 255.0, name='noise_rescale',
    )(inputs)
    noise_model = noise_backbone(noise_normalized.shape[1:])
    noise_features = noise_model(noise_normalized)

    # ── Multistream Feature Fusion ─────────────────────────────────────
    fused = multistream_feature_fusion(
        texture_features, frequency_features, noise_features,
    )
    # fused shape: (batch, 64, 64, 256) for 512x512 input

    # ── Segmentation Head ──────────────────────────────────────────────
    # Texture features serve as encoder skip connections for the decoder.
    # Expected spatial alignment with the segmentation head:
    #   skip1 → Dt2 (128x128)   matched after 1st decoder upsample
    #   skip2 → Dt3 ( 64x64)    matched at bottleneck upsample
    #   skip3 → Dt4 ( 32x32)    matched at bottleneck after stride-2
    Dt1, Dt2, Dt3, Dt4 = texture_features
    encoder_features = [Dt2, Dt3, Dt4]

    mask = segmentation_head(fused, encoder_features)
    # mask shape: (batch, 512, 512, 1) — sigmoid activated

    # ── Build & Compile ────────────────────────────────────────────────
    model = tf.keras.Model(
        inputs=inputs, outputs=mask, name='tampering_detection',
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    model = build_model()
    model.summary()
