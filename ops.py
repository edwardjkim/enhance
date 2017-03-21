"""
Modified from
https://github.com/Tetrachrome/subpixel/blob/master/subpixel.py
"""
import tensorflow as tf


def _phase_shift(I, r):
    batch_size, depth, height, width, channels = I.get_shape().as_list()
    #X = tf.reshape(I, (-1, a, b, r, r))
    X = tf.reshape(I, (-1, depth, height, width, r, r))
    #X = tf.transpose(X, (0, 1, 2, 4, 3))
    X = tf.transpose(X, (0, 1, 2, 3, 5, 4))
    #X = tf.split(X, a, axis=1)
    X = tf.split(X, height, axis=2)
    #X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)
    X = tf.concat([tf.squeeze(x, axis=2) for x in X], axis=3)
    #X = tf.split(X, b, axis=1)
    X = tf.split(X, width, axis=2)
    #X = tf.concat([tf.squeeze(x, axis=1) for x in X], axis=2)
    X = tf.concat([tf.squeeze(x, axis=2) for x in X], axis=3)
    #return tf.reshape(X, (-1, a*r, b*r, 1))
    return tf.reshape(X, (-1, depth, height * r, width * r, 1))


def periodic_shuffle(X, r, color=False):
    if color:
        #Xc = tf.split(X, 3, axis=3)
        Xc = tf.split(X, 3, axis=4)
        #X = tf.concat([_phase_shift(x, r) for x in Xc], axis=3)
        X = tf.concat([_phase_shift(x, r) for x in Xc], axis=4)
    else:
        X = _phase_shift(X, r)
    return X
