from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import glob
import random

import tensorflow as tf

import sres_input
from ops import periodic_shuffle


FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer(
    'batch_size', 64,
    """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string(
    'data_dir', '/notebooks/shared/videos/webcam/tfrecords',
    """Path to the data directory.""")
tf.app.flags.DEFINE_boolean(
    'use_fp16', False,
    """Train the model using fp16.""")
tf.app.flags.DEFINE_integer(
    'upscale_factor', 4,
    """The magnify factor.""")

IMAGE_ROWS = sres_input.IMAGE_ROWS
IMAGE_COLS = sres_input.IMAGE_COLS
NUM_CHANNELS = sres_input.NUM_CHANNELS

# Constants describing the training process.
INITIAL_LEARNING_RATE = 0.0002       # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'


def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):

      dtype = tf.float16 if FLAGS.use_fp16 else tf.float32

      # https://github.com/tensorflow/tensorflow/issues/1317
      try:
          var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
      except:
          with tf.variable_scope(tf.get_variable_scope(), reuse=True):
              var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)

  return var


def _initialized_variable(name, shape, stddev):
  """Helper to create a Variable initialized with a truncated normal distribution.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
  Returns:
    Variable Tensor
  """
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
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

    filenames = glob.glob(os.path.join(FLAGS.data_dir, '*'))
    random.shuffle(filenames)

    images = sres_input.distorted_inputs(filenames=filenames, batch_size=FLAGS.batch_size)
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
    return images


def generator(input_image):

    batch_size = tf.shape(input_image)[0]

    with tf.variable_scope('gen'):
  
        with tf.variable_scope('deconv1'):
            kernel = _initialized_variable('weights', shape=[1, 1, 1, 64, 3], stddev=0.02)
            deconv_shape = [batch_size, 3, 360 // FLAGS.upscale_factor, 640 // FLAGS.upscale_factor, 64]
            conv_t = tf.nn.conv3d_transpose(
                input_image, kernel,
                output_shape=deconv_shape, strides=[1, 1, 1, 1, 1])
            # https://github.com/tensorflow/tensorflow/issues/833
            conv_t = tf.reshape(conv_t, deconv_shape)
            biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv_t, biases)
            # prelu
            alphas = _variable_on_cpu('alpha', [64], tf.constant_initializer(0.2))
            deconv1 = tf.nn.relu(bias) + alphas * (bias - abs(bias)) * 0.5

        with tf.variable_scope('deconv2'):
            kernel = _initialized_variable('weights', shape=[1, 5, 5, 64, 64], stddev=0.02)
            deconv_shape = [batch_size, 3, 360 // FLAGS.upscale_factor, 640 // FLAGS.upscale_factor, 64]
            conv_t = tf.nn.conv3d_transpose(
                deconv1, kernel,
                output_shape=deconv_shape, strides=[1, 1, 1, 1, 1])
            conv_t = tf.reshape(conv_t, deconv_shape)
            biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv_t, biases)
            # prelu
            alphas = _variable_on_cpu('alpha', [64], tf.constant_initializer(0.2))
            deconv2 = tf.nn.relu(bias) + alphas * (bias - abs(bias)) * 0.5

        with tf.variable_scope('deconv3'):
            kernel = _initialized_variable('weights', shape=[1, 5, 5, 3 * FLAGS.upscale_factor ** 2, 64], stddev=0.02)
            deconv_shape = [batch_size, 3, 360 // FLAGS.upscale_factor, 640 // FLAGS.upscale_factor, 3 * FLAGS.upscale_factor ** 2]
            conv_t = tf.nn.conv3d_transpose(
                deconv2, kernel,
                output_shape=deconv_shape, strides=[1, 1, 1, 1, 1])
            conv_t = tf.reshape(conv_t, deconv_shape)
            biases = _variable_on_cpu('biases', [3 * FLAGS.upscale_factor ** 2], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv_t, biases)
            # prelu
            alphas = _variable_on_cpu('alpha', [3 * FLAGS.upscale_factor ** 2], tf.constant_initializer(0.2))
            deconv3 = tf.nn.relu(bias) + alphas * (bias - abs(bias)) * 0.5

        with tf.variable_scope('ps'):
            output = periodic_shuffle(deconv3, FLAGS.upscale_factor, color=True)
      
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
