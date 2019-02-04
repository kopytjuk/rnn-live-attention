from keras import backend as K
import tensorflow as tf


def tpr(y_true, y_pred):

    T_seq = tf.shape(y_true)[1]

    # round to integers
    y_pred_rounded = K.round(y_pred)
    y_true = K.round(y_true)

    y_pred_rounded = K.reshape(y_pred_rounded, shape=(-1, T_seq))
    y_true = K.reshape(y_true, shape=(-1, T_seq))

    T = K.sum(y_true, axis=1)
    TP = K.sum(y_pred_rounded*y_true, axis=1)
    return K.mean(TP/T)

def tnr(y_true, y_pred):
    # round to integers
    y_pred_rounded = K.round(y_pred)
    y_true = K.round(y_true)

    y_pred_rounded = y_pred >= 1
    y_true = y_true >= 1

    T_seq = tf.shape(y_true)[1]

    y_pred_rounded = K.reshape(y_pred_rounded, shape=(-1, T_seq))
    y_true = K.reshape(y_true, shape=(-1, T_seq))

    y_pred_rounded = tf.math.logical_not(y_pred_rounded)
    y_true = tf.math.logical_not(y_true)

    y_true = tf.cast(y_true, "uint8")
    y_pred_rounded = tf.cast(y_pred_rounded, "uint8")

    N = K.sum(y_true, axis=1)
    TN = K.sum(y_pred_rounded*y_true, axis=1)
    return K.mean(TN/N)