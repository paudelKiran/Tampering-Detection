import tensorflow as tf


def rgb_to_ycbcr(image):
    image = tf.cast(image, tf.float32) / 255.0
    matrix = tf.constant([
        [0.299, -0.168736, 0.5],
        [0.587, -0.331264, -0.418688],
        [0.114, 0.5, -0.081312]
    ])

    shift = tf.constant([0.0, 0.5, 0.5])
    ycbcr = tf.tensordot(image, matrix, axes=1) + shift
    return ycbcr


def block_dct(channel):
    B = tf.shape(channel)[0]
    H = tf.shape(channel)[1]
    W = tf.shape(channel)[2]

    channel = channel[:, :H - H % 8, :W - W % 8]

    patches = tf.image.extract_patches(
        images=tf.expand_dims(channel, -1),
        sizes=[1, 8, 8, 1],
        strides=[1, 8, 8, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    patches = tf.reshape(patches, [-1, 8, 8])
    dct = tf.signal.dct(patches, type=2, norm='ortho')
    dct = tf.signal.dct(tf.transpose(dct, perm=[0, 2, 1]), type=2, norm='ortho')
    dct = tf.transpose(dct, perm=[0, 2, 1])

    return dct


def reconstruct_dct_volume(dct_blocks, H, W):
    blocks_per_image = (H // 8) * (W // 8)
    dct_blocks = tf.reshape(dct_blocks, [-1, blocks_per_image, 8, 8])
    dct_volume = tf.reshape(dct_blocks, [-1, H // 8, W // 8, 64])
    return dct_volume


def simulate_quantization(dct_volume, q=10.0):
    return tf.round(dct_volume / q)


def binary_volume_encoding(dct_volume, thresholds=(-20, -10, 0, 10, 20)):
    binary_maps = []
    for t in thresholds:
        binary_maps.append(tf.cast(dct_volume > t, tf.float32))
    return tf.concat(binary_maps, axis=-1)


def _residual_block(x, filters, name):
    shortcut = x

    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False, name=f'{name}_conv1')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = tf.keras.layers.ReLU(name=f'{name}_relu1')(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding='same', use_bias=False, name=f'{name}_conv2')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name}_bn2')(x)

    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False, name=f'{name}_proj_conv')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(name=f'{name}_proj_bn')(shortcut)

    x = tf.keras.layers.Add(name=f'{name}_add')([x, shortcut])
    x = tf.keras.layers.ReLU(name=f'{name}_out')(x)
    return x


def _fuse_to_target(branches, target_idx, target_filters, stage_name):
    fused = []
    for source_idx, source in enumerate(branches):
        x = source

        if source_idx > target_idx:
            for step in range(source_idx - target_idx):
                x = tf.keras.layers.UpSampling2D(
                    size=2,
                    interpolation='bilinear',
                    name=f'{stage_name}_up_{source_idx}_{target_idx}_{step}'
                )(x)

            x = tf.keras.layers.Conv2D(
                target_filters,
                1,
                padding='same',
                use_bias=False,
                name=f'{stage_name}_up_proj_{source_idx}_{target_idx}'
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f'{stage_name}_up_proj_bn_{source_idx}_{target_idx}')(x)

        elif source_idx < target_idx:
            for step in range(target_idx - source_idx):
                step_filters = target_filters if step == (target_idx - source_idx - 1) else x.shape[-1]
                x = tf.keras.layers.Conv2D(
                    step_filters,
                    3,
                    strides=2,
                    padding='same',
                    use_bias=False,
                    name=f'{stage_name}_down_{source_idx}_{target_idx}_{step}'
                )(x)
                x = tf.keras.layers.BatchNormalization(name=f'{stage_name}_down_bn_{source_idx}_{target_idx}_{step}')(x)
                x = tf.keras.layers.ReLU(name=f'{stage_name}_down_relu_{source_idx}_{target_idx}_{step}')(x)

        else:
            if x.shape[-1] != target_filters:
                x = tf.keras.layers.Conv2D(
                    target_filters,
                    1,
                    padding='same',
                    use_bias=False,
                    name=f'{stage_name}_same_proj_{source_idx}'
                )(x)
                x = tf.keras.layers.BatchNormalization(name=f'{stage_name}_same_proj_bn_{source_idx}')(x)

        fused.append(x)

    x = tf.keras.layers.Add(name=f'{stage_name}_sum_{target_idx}')(fused)
    x = tf.keras.layers.ReLU(name=f'{stage_name}_relu_{target_idx}')(x)
    return x


def _hrnet_stage(branches, filters, blocks, stage_name):
    processed = []
    for idx, x in enumerate(branches):
        for block_idx in range(blocks):
            x = _residual_block(x, filters[idx], name=f'{stage_name}_b{idx}_r{block_idx}')
        processed.append(x)

    fused = []
    for target_idx in range(len(processed)):
        fused.append(_fuse_to_target(processed, target_idx, filters[target_idx], stage_name=stage_name))
    return fused


def frequency_backbone_hrnet_w18(input_shape=(None, None, 320), blocks_per_stage=2):

    inputs = tf.keras.Input(shape=input_shape)

    # Stem
    x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False, name='stem_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name='stem_bn')(x)
    x = tf.keras.layers.ReLU(name='stem_relu')(x)


    for i in range(4):   # HRNet-W18 uses 4 bottlenecks here
        x = _residual_block(x, 64, name=f'stage1_block{i}')

 
    x1 = tf.keras.layers.Conv2D(18, 3, padding='same', use_bias=False, name='transition2_1')(x)
    x1 = tf.keras.layers.BatchNormalization(name='transition2_1_bn')(x1)
    x1 = tf.keras.layers.ReLU()(x1)

    x2 = tf.keras.layers.Conv2D(36, 3, strides=2, padding='same', use_bias=False, name='transition2_2')(x)
    x2 = tf.keras.layers.BatchNormalization(name='transition2_2_bn')(x2)
    x2 = tf.keras.layers.ReLU()(x2)

    x1, x2 = _hrnet_stage([x1, x2], filters=(18, 36), blocks=blocks_per_stage, stage_name='stage2')

    x3 = tf.keras.layers.Conv2D(72, 3, strides=2, padding='same', use_bias=False, name='transition3_3')(x2)
    x3 = tf.keras.layers.BatchNormalization(name='transition3_3_bn')(x3)
    x3 = tf.keras.layers.ReLU()(x3)

 
    x1, x2, x3 = _hrnet_stage([x1, x2, x3], filters=(18, 36, 72), blocks=blocks_per_stage, stage_name='stage3')

    x4 = tf.keras.layers.Conv2D(144, 3, strides=2, padding='same', use_bias=False, name='transition4_4')(x3)
    x4 = tf.keras.layers.BatchNormalization(name='transition4_4_bn')(x4)
    x4 = tf.keras.layers.ReLU()(x4)

    x1, x2, x3, x4 = _hrnet_stage(
        [x1, x2, x3, x4],
        filters=(18, 36, 72, 144),
        blocks=blocks_per_stage,
        stage_name='stage4'
    )

    return tf.keras.Model(inputs, [x1, x2, x3, x4], name='frequency_hrnet_w18')


def frequency_branch_pipeline(image):
    H = tf.shape(image)[1]
    W = tf.shape(image)[2]

    ycbcr = rgb_to_ycbcr(image)
    Y = ycbcr[..., 0]

    dct_blocks = block_dct(Y)
    dct_volume = reconstruct_dct_volume(dct_blocks, H, W)

    dct_volume = simulate_quantization(dct_volume)
    binary_volume = binary_volume_encoding(dct_volume)

    model = frequency_backbone_hrnet_w18(binary_volume.shape[1:])
    features = model(binary_volume)
    return features