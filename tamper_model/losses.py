import tensorflow as tf


CLIP_MIN = 1e-7
CLIP_MAX = 1.0 - 1e-7
SMOOTH = 1.0


def dice_loss(y_true, y_pred):

    y_pred = tf.clip_by_value(y_pred, CLIP_MIN, CLIP_MAX)
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    return 1.0 - (2.0 * intersection + SMOOTH) / (denominator + SMOOTH)


def bce_loss(y_true, y_pred):

    y_pred = tf.clip_by_value(y_pred, CLIP_MIN, CLIP_MAX)
    y_true = tf.cast(y_true, tf.float32)

    bce = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(bce)


def combined_loss(y_true, y_pred):

    return bce_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


def dice_coefficient(y_true, y_pred):

    y_pred = tf.clip_by_value(y_pred, CLIP_MIN, CLIP_MAX)
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    return (2.0 * intersection + SMOOTH) / (denominator + SMOOTH)


def get_loss_function():

    return combined_loss
