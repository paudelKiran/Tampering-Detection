import tensorflow as tf


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


def hrnet_w18_backbone(input_shape=(None, None, 3), blocks_per_stage=2, name='hrnet_w18'):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False, name='stem_conv')(inputs)
    x = tf.keras.layers.BatchNormalization(name='stem_bn')(x)
    x = tf.keras.layers.ReLU(name='stem_relu')(x)

    for i in range(4):
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

    return tf.keras.Model(inputs, [x1, x2, x3, x4], name=name)


__all__ = ['hrnet_w18_backbone']
