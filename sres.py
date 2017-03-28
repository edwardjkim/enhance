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
    'data_dir', '/notebooks/shared/videos/webcam',
    """Path to the data directory.""")
tf.app.flags.DEFINE_boolean(
    'use_fp16', False,
    """Train the model using fp16.""")
tf.app.flags.DEFINE_integer(
    'upscale_factor', 4,
    """The magnify factor.""")
tf.app.flags.DEFINE_float(
    'initial_learning_rate', 0.0001,
    """The initial learning rate.""")
tf.app.flags.DEFINE_float(
    'adam_momentum', 0.5,
    """The beta1 momentum in Adam optimizer.""")
tf.app.flags.DEFINE_integer(
    'num_filters', 64,
    """Number of filters in the convolutional layers.""")
tf.app.flags.DEFINE_float(
    'initial_weights_stddev', 0.02,
    """The standard deviation of the truncated Gaussian for convolutional kernels.""")
tf.app.flags.DEFINE_integer(
    'first_filter_size', 5,
    """Size of filters in the first convolutional layer.""")
tf.app.flags.DEFINE_integer(
    'second_filter_size', 3,
    """Size of filters in the second convolutional layer.""")
tf.app.flags.DEFINE_integer(
    'third_filter_size', 3,
    """Size of filters in the third convolutional layer.""")


IMAGE_HEIGHT = sres_input.IMAGE_HEIGHT
IMAGE_WIDTH = sres_input.IMAGE_WIDTH
NUM_CHANNELS = sres_input.NUM_CHANNELS

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = sres_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = sres_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
INITIAL_LEARNING_RATE = FLAGS.initial_learning_rate       # Initial learning rate.
ADAM_MOMENTUM = FLAGS.adam_momentum
NUM_FILTERS = FLAGS.num_filters
INITIAL_WEIGHTS_STDDEV = FLAGS.initial_weights_stddev
FIRST_FILTER_SIZE = FLAGS.first_filter_size
SECOND_FILTER_SIZE = FLAGS.second_filter_size
THIRD_FILTER_SIZE = FLAGS.third_filter_size

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/gpu:0'):

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


def inputs(eval_data=False):
    """Construct input for training using the Reader ops.
  
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.

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

    if eval_data:
        filenames = glob.glob(os.path.join(FLAGS.data_dir, 'valid', '*'))
    else:
        filenames = glob.glob(os.path.join(FLAGS.data_dir, 'train', '*'))
        random.shuffle(filenames)

    images = sres_input.inputs(
        filenames=filenames, batch_size=FLAGS.batch_size, shuffle=eval_data)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)

    return images


def generator(input_image):

    batch_size = tf.shape(input_image)[0]

    with tf.variable_scope('gen'):
  
        with tf.variable_scope('deconv1'):
            kernel = _initialized_variable(
                'weights', shape=[1, FIRST_FILTER_SIZE, FIRST_FILTER_SIZE, NUM_FILTERS, 3], stddev=INITIAL_WEIGHTS_STDDEV)
            deconv_shape = [
                batch_size,
                2,
                IMAGE_HEIGHT // FLAGS.upscale_factor,
                IMAGE_WIDTH // FLAGS.upscale_factor,
                NUM_FILTERS]
            conv_t = tf.nn.conv3d_transpose(
                input_image, kernel,
                output_shape=deconv_shape, strides=[1, 1, 1, 1, 1])
            # https://github.com/tensorflow/tensorflow/issues/833
            conv_t = tf.reshape(conv_t, deconv_shape)
            biases = _variable_on_cpu('biases', [NUM_FILTERS], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv_t, biases)
            # prelu
            alphas = _variable_on_cpu('alpha', [NUM_FILTERS], tf.constant_initializer(0.2))
            deconv1 = tf.nn.relu(bias) + alphas * (bias - abs(bias)) * 0.5

        with tf.variable_scope('deconv2'):
            kernel = _initialized_variable('weights', shape=[1, SECOND_FILTER_SIZE, SECOND_FILTER_SIZE, NUM_FILTERS, NUM_FILTERS], stddev=INITIAL_WEIGHTS_STDDEV)
            deconv_shape = [
                batch_size,
                2,
                IMAGE_HEIGHT // FLAGS.upscale_factor,
                IMAGE_WIDTH // FLAGS.upscale_factor,
                NUM_FILTERS]
            conv_t = tf.nn.conv3d_transpose(
                deconv1, kernel,
                output_shape=deconv_shape, strides=[1, 1, 1, 1, 1])
            conv_t = tf.reshape(conv_t, deconv_shape)
            biases = _variable_on_cpu('biases', [NUM_FILTERS], tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv_t, biases)
            # prelu
            alphas = _variable_on_cpu('alpha', [NUM_FILTERS], tf.constant_initializer(0.2))
            deconv2 = tf.nn.relu(bias) + alphas * (bias - abs(bias)) * 0.5

        with tf.variable_scope('deconv3'):
            kernel = _initialized_variable(
                'weights',
                shape=[1, THIRD_FILTER_SIZE, THIRD_FILTER_SIZE, 3 * FLAGS.upscale_factor ** 2, NUM_FILTERS],
                stddev=INITIAL_WEIGHTS_STDDEV)
            deconv_shape = [
                batch_size,
                2,
                IMAGE_HEIGHT // FLAGS.upscale_factor,
                IMAGE_WIDTH // FLAGS.upscale_factor,
                3 * FLAGS.upscale_factor ** 2]
            conv_t = tf.nn.conv3d_transpose(
                deconv2, kernel,
                output_shape=deconv_shape, strides=[1, 1, 1, 1, 1])
            conv_t = tf.reshape(conv_t, deconv_shape)
            biases = _variable_on_cpu(
                'biases',
                [3 * FLAGS.upscale_factor ** 2],
                tf.constant_initializer(0.0))
            bias = tf.nn.bias_add(conv_t, biases)
            # prelu
            alphas = _variable_on_cpu(
                'alpha',
                [3 * FLAGS.upscale_factor ** 2],
                tf.constant_initializer(0.2))
            deconv3 = tf.nn.relu(bias) + alphas * (bias - abs(bias)) * 0.5

        with tf.variable_scope('ps'):
            output = periodic_shuffle(
                deconv3, FLAGS.upscale_factor, color=True)
      
    return tf.nn.tanh(output)


def loss(real, fake):
    """
    """
    mse = tf.reduce_mean(tf.square(tf.subtract(real, fake)))
    tf.add_to_collection('losses', mse)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def valid_loss(real, fake):
    """
    """
    mse = tf.reduce_mean(tf.square(tf.subtract(real, fake)))
    tf.add_to_collection('valid_losses', mse)

    return tf.add_n(tf.get_collection('valid_losses'), name='total_valid_loss')


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
        opt = tf.train.AdamOptimizer(INITIAL_LEARNING_RATE, beta1=ADAM_MOMENTUM)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
        apply_op = opt.minimize(total_loss, var_list=var_list)
  
    with tf.control_dependencies([apply_op]):
        train_op = tf.no_op(name='train')
  
    return train_op
