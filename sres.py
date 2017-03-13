from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys

import tensorflow as tf

import sres_input
from ops import PS


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer(
    'batch_size', 64,
    """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string(
    'data_dir', '/notebooks/shared/videos/youtube/tfrecords',
    """Path to the data directory.""")
tf.app.flags.DEFINE_boolean(
    'use_fp16', False,
    """Train the model using fp16.""")

IMAGE_ROWS = sres_input.IMAGE_ROWS
IMAGE_COLS = sres_input.IMAGE_COLS
NUM_CHANNELS = sres_input.NUM_CHANNELS

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = sres_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = sres_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
INITIAL_LEARNING_RATE = 0.0004       # Initial learning rate.


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
  
    Parameters
    ----------
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  
    Returns
    -------
    Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
  
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
  
    Parameters
    ----------
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  
    Returns
    -------
    Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs():
    """Construct distorted input for training using the Reader ops.
  
    Returns
    -------
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  
    Raises
    ------
    ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    images = sres_input.distorted_inputs(data_dir=data_dir, batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
    return images


def inputs(eval_data):
    """Construct input for evaluation using the Reader ops.
  
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
  
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
  
    Raises:
      ValueError: If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'test.tfrecords')
    images = sres_input.inputs(
        eval_data=eval_data,
        data_dir=data_dir,
        batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
    return images


def generator(input_image):

    with tf.variable_scope('gen'):
  
        with tf.variable_scope('deconv1'):
            kernel = _variable_with_weight_decay(
              'weights', shape=[1, 1, 64, 3], stddev=0.02, wd=None)
            conv_t = tf.nn.conv2d_transpose(input_image, kernel, output_shape=[FLAGS.batch_size, 180, 320, 64], strides=[1, 1, 1, 1])
            biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv_t, biases)
            deconv1 = tf.maximum(bias, 0.2 * bias) # leaky relu
      
        print("deconv1: ", deconv1.get_shape())
    
        with tf.variable_scope('deconv2'):
            kernel = _variable_with_weight_decay(
              'weights', shape=[5, 5, 64, 64], stddev=0.02, wd=None)
            conv_t = tf.nn.conv2d_transpose(deconv1, kernel, output_shape=[FLAGS.batch_size, 180, 320, 64], strides=[1, 1, 1, 1])
            biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv_t, biases)
            deconv2 = tf.maximum(bias, 0.2 * bias) # leaky relu
      
        print("deconv2: ", deconv2.get_shape())
    
        with tf.variable_scope('deconv3'):
            kernel = _variable_with_weight_decay(
              'weights', shape=[5, 5, 3 * 4, 64], stddev=0.02, wd=None)
            conv_t = tf.nn.conv2d_transpose(deconv2, kernel, output_shape=[FLAGS.batch_size, 180, 320, 3 * 4], strides=[1, 1, 1, 1])
            biases = _variable_on_cpu('biases', [3 * 4], tf.constant_initializer(0.0))
            deconv3 = tf.nn.bias_add(conv_t, biases)

        print("deconv3: ", deconv3.get_shape())

        with tf.variable_scope('ps'):
            output = PS(deconv3, 2, color=True)
      
        print("output: ", output.get_shape())
  
    return tf.nn.tanh(output)


def loss(real, fake):
    """
    """
    mse = tf.reduce_mean(tf.square(tf.subtract(real, fake)))
    tf.add_to_collection('losses', mse)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(total_loss):
    """Train model.
  
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
  
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Compute gradients.
    with tf.control_dependencies([total_loss]):
        opt = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE, beta1=0.5)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
        apply_op = opt.minimize(total_loss, var_list=var_list)
  
    with tf.control_dependencies([apply_op]):
        train_op = tf.no_op(name='train')
  
    return train_op
