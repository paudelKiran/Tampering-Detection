import tensorflow as tf
import numpy as np

layers = tf.keras.layers

MSIZE = 24
M = np.zeros((MSIZE, MSIZE), dtype=np.float32)
M[2:-2, 2:-2] = 1.0


class LocalAttentionLayer(tf.keras.layers.Layer):
    """Applies local attention: Conv1x1 → BN → ELU+1, multiplied by resized mask M."""

    def __init__(self, out_channels=4, **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.conv = layers.Conv2D(out_channels, 1, padding="same", use_bias=False)
        self.bn = layers.BatchNormalization()
        self.M_init = M.copy()

    def build(self, input_shape):
        super().build(input_shape)
        self.M_tensor = tf.constant(
            self.M_init[np.newaxis, :, :, np.newaxis], dtype=tf.float32
        )

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = tf.keras.activations.elu(x) + 1.0

        target_h = tf.shape(x)[1]
        target_w = tf.shape(x)[2]
        M1 = tf.image.resize(self.M_tensor, [target_h, target_w], method='bilinear')
        M1 = tf.cast(M1, x.dtype)
        return x * M1

    def get_config(self):
        config = super().get_config()
        config['out_channels'] = self.out_channels
        return config


def l1(inputs):
    x = layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False, name='l1_conv')(inputs)
    x = layers.BatchNormalization(name='l1_batch')(x)
    x = layers.ReLU(name='l1_relu')(x)
    return x


def block1(img):
    l1_img = img
    x = layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False, name='b1_conv1')(img)
    x = layers.BatchNormalization(name='b1_batch1')(x)
    x = layers.ReLU(name='b1_relu1')(x)

    x = layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False, name='b1_conv2')(x)
    x = layers.BatchNormalization(name='b1_batch2')(x)
    x = layers.Add(name="b1_add")([x, l1_img])
    x = layers.ReLU(name='b1_relu2')(x)
    return x


def local_attention(img):
    return LocalAttentionLayer(out_channels=4, name='local_attention')(img)


def difference_feature(l1_out):
    l2 = layers.AveragePooling2D(pool_size=4, strides=4, padding='same', name="l2_avgpool")(l1_out)
    l2_up = layers.UpSampling2D(size=4, interpolation='bilinear', name='l2_upsample')(l2)
    T_L = layers.Subtract(name='T_L')([l1_out, l2_up])
    return T_L


def block2(x):
    d1 = layers.Conv2D(64, 3, padding='same', use_bias=False, name='b2_conv1')(x)
    d1 = layers.BatchNormalization(name='b2_batch1')(d1)
    d1 = layers.ReLU(name='b2_relu1')(d1)

    d2_input = layers.Concatenate(name='b2_concat1')([x, d1])
    d2 = layers.Conv2D(64, 3, padding='same', use_bias=False, name='b2_conv2')(d2_input)
    d2 = layers.BatchNormalization(name='b2_batch2')(d2)
    d2 = layers.ReLU(name='b2_relu2')(d2)

    d3_input = layers.Concatenate(name='b2_concat2')([x, d1, d2])
    d3 = layers.Conv2D(64, 3, padding='same', use_bias=False, name='b2_conv3')(d3_input)
    d3 = layers.BatchNormalization(name='b2_batch3')(d3)
    d3 = layers.ReLU(name='b2_relu3')(d3)

    E = layers.Concatenate(name='b2_concat_final')([x, d1, d2, d3])
    E = layers.Conv2D(256, 1, padding='same', use_bias=False, name='b2_conv_final')(E)
    E = layers.BatchNormalization(name='b2_batchf')(E)
    E = layers.ReLU(name='b2_reluf')(E)
    return E


def texture_stream(inputs):

    
    x = layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False,
                       name='texture_stem_conv')(inputs)
    x = layers.BatchNormalization(name='texture_stem_bn')(x)
    x = layers.ReLU(name='texture_stem_relu')(x)

    # L1: general feature extraction
    l1_out = l1(x)

    # Block 1: residual refinement
    h1 = block1(l1_out)

    # Local attention
    t_h = local_attention(h1)

    # Difference feature
    t_l = difference_feature(l1_out)

   
    t_h_up = layers.Conv2D(64, 1, padding='same', use_bias=False, name='attn_project_conv')(t_h)
    t_h_up = layers.BatchNormalization(name='attn_project_bn')(t_h_up)
    t_h_up = layers.ReLU(name='attn_project_relu')(t_h_up)
    enhanced = layers.Multiply(name='texture_enhance_multiply')([t_h_up, t_l])

    
    E = block2(enhanced)

    return E
