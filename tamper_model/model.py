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
        fuse_noise,
        final_fusion,
    )
    from tamper_model.segmentation import segmentation_head
    from tamper_model.classification import classification_head
    from tamper_model.losses import (
        segmentation_loss,
        classification_loss,
        dice_coefficient,
        get_loss_functions,
    )
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
    from feature_fusion import fuse_noise, final_fusion
    from segmentation import segmentation_head
    from classification import classification_head
    from losses import (
        segmentation_loss,
        classification_loss,
        dice_coefficient,
        get_loss_functions,
    )



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


def multistream_feature_fusion(texture_feat, frequency_features, noise_features):

    Dn1, Dn2, Dn3, Dn4 = noise_features

    # Resize frequency features to match texture spatial dims
    tex_h, tex_w = texture_feat.shape[1], texture_feat.shape[2]
    freq_resized = [
        tf.keras.layers.Resizing(
            tex_h, tex_w, interpolation='bilinear',
            name=f'freq_resize_{i}',
        )(f)
        for i, f in enumerate(frequency_features)
    ]
    freq_concat = tf.keras.layers.Concatenate(
        axis=-1, name='freq_concat',
    )(freq_resized)

    # Fuse texture + frequency
    Dtf = tf.keras.layers.Concatenate(
        axis=-1, name='texture_freq_concat',
    )([texture_feat, freq_concat])
    Dtf = tf.keras.layers.Conv2D(
        256, 3, padding='same', use_bias=False, name='texture_freq_proj',
    )(Dtf)
    Dtf = tf.keras.layers.BatchNormalization(name='texture_freq_bn')(Dtf)
    Dtf = tf.keras.layers.ReLU(name='texture_freq_relu')(Dtf)

    # Fuse noise features
    Dn_fused = fuse_noise(Dn1, Dn2, Dn3, Dn4)

    Dn_matched = tf.keras.layers.Resizing(
        Dtf.shape[1], Dtf.shape[2], interpolation='bilinear',
    )(Dn_fused)

    fused = final_fusion(Dn_matched, Dtf)

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




def build_model(input_shape=(128, 128, 3)):

    inputs = tf.keras.Input(shape=input_shape, name='input_image')


    texture_feat = texture_stream(inputs)  # (h/2, w/2, 256)

  
    freq_preprocessed = FrequencyPreprocessLayer(
        name='frequency_preprocess',
    )(inputs)
    freq_backbone = frequency_backbone_hrnet_w18()
    frequency_features = freq_backbone(freq_preprocessed)


    noise_normalized = tf.keras.layers.Rescaling(
        1.0 / 255.0, name='noise_rescale',
    )(inputs)
    noise_model = noise_backbone(noise_normalized.shape[1:])
    noise_features = noise_model(noise_normalized)


    fused = multistream_feature_fusion(
        texture_feat, frequency_features, noise_features,
    )

    # Generate multi-scale skip connections from texture feature
    skip1 = tf.keras.layers.Conv2D(
        64, 3, strides=2, padding='same', use_bias=False,
        name='texture_skip1_conv',
    )(texture_feat)
    skip1 = tf.keras.layers.BatchNormalization(name='texture_skip1_bn')(skip1)
    skip1 = tf.keras.layers.ReLU(name='texture_skip1_relu')(skip1)

    skip2 = tf.keras.layers.Conv2D(
        128, 3, strides=2, padding='same', use_bias=False,
        name='texture_skip2_conv',
    )(skip1)
    skip2 = tf.keras.layers.BatchNormalization(name='texture_skip2_bn')(skip2)
    skip2 = tf.keras.layers.ReLU(name='texture_skip2_relu')(skip2)

    skip3 = tf.keras.layers.Conv2D(
        256, 3, strides=2, padding='same', use_bias=False,
        name='texture_skip3_conv',
    )(skip2)
    skip3 = tf.keras.layers.BatchNormalization(name='texture_skip3_bn')(skip3)
    skip3 = tf.keras.layers.ReLU(name='texture_skip3_relu')(skip3)

    encoder_features = [skip1, skip2, skip3]

    mask = segmentation_head(fused, encoder_features)

    # Image-level binary authenticity prediction
    cls_output = classification_head(fused)

    model = tf.keras.Model(
        inputs=inputs,
        outputs={'segmentation': mask, 'classification': cls_output},
        name='tampering_detection',
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=get_loss_functions(),
        metrics={
            'segmentation': [dice_coefficient, 'accuracy'],
            'classification': ['accuracy'],
        },
    )

    return model


if __name__ == '__main__':
    model = build_model()
    model.summary()
