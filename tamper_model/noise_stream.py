import tensorflow as tf

try:
    from tamper_model.bayerconv_layer import BayarConv
except ImportError:
    from bayerconv_layer import BayarConv


def noise_backbone(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # BayarConv preprocessing
    x = BayarConv(filters=3)(inputs)

    # Feature extraction (no pooling)
    x1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    x2 = tf.keras.layers.Conv2D(
        128, 3, strides=2, padding='same', activation='relu')(x1)

    x3 = tf.keras.layers.Conv2D(
        256, 3, strides=2, padding='same', activation='relu')(x2)

    x4 = tf.keras.layers.Conv2D(
        512, 3, strides=2, padding='same', activation='relu')(x3)

    return tf.keras.Model(inputs, [x1, x2, x3, x4])
    
    

def noise_branch_pipeline(image):

    image = tf.cast(image, tf.float32) / 255.0

    model = noise_backbone(image.shape[1:])
    features = model(image)

    return features




