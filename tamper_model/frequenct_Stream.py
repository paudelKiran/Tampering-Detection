import tensorflow as tf

try:
    from tamper_model.hrnet_backbone import hrnet_w18_backbone
except ImportError:
    from hrnet_backbone import hrnet_w18_backbone


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


def frequency_backbone_hrnet_w18(input_shape=(None, None, 320), blocks_per_stage=2):
    return hrnet_w18_backbone(
        input_shape=input_shape,
        blocks_per_stage=blocks_per_stage,
        name='frequency_hrnet_w18'
    )


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