import tensorflow as tf



def conv_block(x, filters):

    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x



def decoder_block(x, skip, filters):

    x = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation="bilinear"
    )(x)

  
    x = tf.keras.layers.Concatenate(axis=-1)([x, skip])

    
    x = conv_block(x, filters)

    return x



def segmentation_head(fused_features, encoder_features):
    skip1, skip2, skip3 = encoder_features  # unpack

   
    bottleneck = tf.keras.layers.Conv2D(
        256, 3, strides=2, padding="same", use_bias=False
    )(fused_features)
    bottleneck = tf.keras.layers.BatchNormalization()(bottleneck)
    bottleneck = tf.keras.layers.ReLU()(bottleneck)

    bottleneck = tf.keras.layers.Concatenate(axis=-1)([bottleneck, skip3])
    bottleneck = conv_block(bottleneck, 256)

    
    x = decoder_block(bottleneck, skip2, 128)

    
    x = decoder_block(x, skip1, 64)

   
    x = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation="bilinear"
    )(x)
    x = conv_block(x, 32)

    
    x = tf.keras.layers.UpSampling2D(
        size=(2, 2), interpolation="bilinear"
    )(x)
    x = conv_block(x, 16)

    
    tampering_mask = tf.keras.layers.Conv2D(
        1, kernel_size=1, padding="same", activation="sigmoid"
    )(x)

    return tampering_mask





   
