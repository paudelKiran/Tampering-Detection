import tensorflow as tf

def upsample(x, scale=2):
    return tf.keras.layers.UpSampling2D(size=(scale,scale), interpolation='bilinear')(x)

def fuse_noise(Dn1, Dn2, Dn3, Dn4):

    x = upsample(Dn4)
    x = tf.concat([x, Dn3], axis=-1)

    x = upsample(x)
    x = tf.concat([x, Dn2], axis=-1)

    x = upsample(x)
    x = tf.concat([x, Dn1], axis=-1)

    return x


def fuse_texture_frequency(Dt1, Dt2, Dt3, Dt4,
                           Df2, Df3, Df4):

    Dtf2 = tf.concat([Dt2, Df2], axis=-1)
    Dtf3 = tf.concat([Dt3, Df3], axis=-1)
    Dtf4 = tf.concat([Dt4, Df4], axis=-1)

    x = upsample(Dtf4)
    x = tf.concat([x, Dtf3], axis=-1)

    x = upsample(x)
    x = tf.concat([x, Dtf2], axis=-1)

    x = upsample(x)
    x = tf.concat([x, Dt1], axis=-1)

    return x


def final_fusion(Dn, Dtf):

    fused = tf.concat([Dn, Dtf], axis=-1)

    fused = tf.keras.layers.Conv2D(
        256, 3, padding='same', activation='relu')(fused)

    return fused