"""
Modified from
https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py
"""
import tensorflow as tf


def _phase_shift(I, r):
    bsize, a, b, c = I.get_shape().as_list()
    bsize = tf.shape(I)[0]
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))
    X = tf.split(X, a, axis=1)
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)
    X = tf.split(X, b, axis=1)
    X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)
    return tf.reshape(X, (bsize, a*r, b*r, 1))


def PS(X, r, color=False):
    if color:
        Xc = tf.split(X, 3, axis=3)
        X = tf.concat([_phase_shift(x, r) for x in Xc], axis=3)
    else:
        X = _phase_shift(X, r)
    return X
