import tensorflow as tf

def upsample(x, scale=2):
    return tf.keras.layers.UpSampling2D(size=(scale,scale), interpolation='bilinear')(x)

def fuse_noise(Dn1, Dn2, Dn3, Dn4):

    x = upsample(Dn4)
    x = tf.keras.layers.Concatenate(axis=-1)([x, Dn3])

    x = upsample(x)
    x = tf.keras.layers.Concatenate(axis=-1)([x, Dn2])

    x = upsample(x)
    x = tf.keras.layers.Concatenate(axis=-1)([x, Dn1])

    return x


def fuse_texture_frequency(Dt1, Dt2, Dt3, Dt4,
                           Df2, Df3, Df4):

    Dtf2 = tf.keras.layers.Concatenate(axis=-1)([Dt2, Df2])
    Dtf3 = tf.keras.layers.Concatenate(axis=-1)([Dt3, Df3])
    Dtf4 = tf.keras.layers.Concatenate(axis=-1)([Dt4, Df4])

    x = upsample(Dtf4)
    x = tf.keras.layers.Concatenate(axis=-1)([x, Dtf3])

    x = upsample(x)
    x = tf.keras.layers.Concatenate(axis=-1)([x, Dtf2])

    x = upsample(x)
    x = tf.keras.layers.Concatenate(axis=-1)([x, Dt1])

    return x


def final_fusion(Dn, Dtf):

    fused = tf.keras.layers.Concatenate(axis=-1)([Dn, Dtf])

    fused = tf.keras.layers.Conv2D(
        256, 3, padding='same', activation='relu')(fused)

    return fused