"""
Modified from
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import sys
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import sres


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/sres_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 1,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


IMAGE_HEIGHT = sres.IMAGE_HEIGHT
IMAGE_WIDTH = sres.IMAGE_WIDTH
NUM_CHANNELS = sres.NUM_CHANNELS

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations.
TOWER_NAME = 'tower'


def tower_loss(scope):
    """Calculate the total loss on a single tower running the model.
  
    Args:
      scope: unique prefix string identifying the tower, e.g. 'tower_0'
  
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # Get images
    real_images = sres.inputs()
    downsampled_real_images = split_and_resize(real_images)
  
    # Build inference Graph.
    fake_images = sres.generator(downsampled_real_images)
  
    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = sres.loss(real_images, fake_images)
  
    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('losses', scope)
  
    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')
  
    return total_loss


def tower_valid_loss(scope):
    """Calculate the total loss on a single tower running the model.
  
    Args:
      scope: unique prefix string identifying the tower, e.g. 'tower_0'
  
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # Get images
    real_images = sres.inputs(eval_data=True)
    downsampled_real_images = split_and_resize(real_images)
  
    # Build inference Graph.
    fake_images = sres.generator(downsampled_real_images)
  
    # Build the portion of the Graph calculating the losses. Note that we will
    # assemble the total_loss using a custom function below.
    _ = sres.valid_loss(real_images, fake_images)
  
    # Assemble all of the losses for the current tower only.
    losses = tf.get_collection('valid_losses', scope)
  
    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_valid_loss')
  
    return total_loss


def average_gradients(tower_grads):
    """
    Calculate the average gradient for each shared variable across all
    towers.
  
    Note that this function provides a synchronization point across all towers.
  
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been
       averaged across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)
  
        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)
  
      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)
  
      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads


def split_and_resize(images):

    images = tf.split(images, 2, axis=1)

    downsampled_images = []
    for i in range(2):
        image = tf.squeeze(images[i], squeeze_dims=[1])
        downsampled_height = int(IMAGE_HEIGHT / FLAGS.upscale_factor)
        downsampled_width = int(IMAGE_WIDTH / FLAGS.upscale_factor)
        downsampled_images.append(tf.image.resize_images(
            image,
            [downsampled_height, downsampled_width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
    downsampled_images = tf.stack(downsampled_images)
    downsampled_images = tf.transpose(
        downsampled_images, perm=[1, 0, 2, 3, 4])

    return downsampled_images


def train():

    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls.
        # This equals the number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
  
        # Create an optimizer that performs gradient descent.
        opt = tf.train.AdamOptimizer(
            sres.INITIAL_LEARNING_RATE, beta1=sres.ADAM_MOMENTUM)
  
        # Calculate the gradients for each model tower.
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                    # Calculate the loss for one tower of the model. This
                    # function constructs the entire model but shares the
                    # variables across all towers.
                        loss = tower_loss(scope)
  
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
  
                        # Calculate the gradients for the batch of data on this
                        # tower.
                        grads = opt.compute_gradients(loss)
  
                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
  
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)
  
        # Calculate the losses for each model tower.
        tower_valid_losses = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                    # Calculate the loss for one tower of the model. This
                    # function constructs the entire model but shares the
                    # variables across all towers.
                        valid_loss = tower_valid_loss(scope)
  
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()
  
                        # Keep track of the gradients across all towers.
                        tower_valid_losses.append(valid_loss)
  
        total_valid_loss = tf.reduce_mean(tower_valid_losses)
  
        # Apply the gradients to adjust the shared variables.
        train_op = opt.apply_gradients(grads, global_step=global_step)
  
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())
  
        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()
  
        # Start running operations on the Graph. allow_soft_placement must be
        # set to True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
  
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
  
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
  
        best_valid_loss = np.inf
        early_stopping_rounds = 0
  
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time
    
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
    
            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus
      
                format_str = ('%s: step %d, loss = %.6f '
                    '(%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                    examples_per_sec, sec_per_batch))
                sys.stdout.flush()
    
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:

                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                current_loss = 0
                num_steps_per_eval = int(
                    sres.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / FLAGS.batch_size /
                    FLAGS.num_gpus)
                for _ in range(num_steps_per_eval):
                    current_loss += sess.run(total_valid_loss)
                current_loss /= num_steps_per_eval

                format_str = ('Validation: step %d, validation loss = %.6f')
                print(format_str % (step, current_loss))
                sys.stdout.flush()

                if current_loss < best_valid_loss:
                    best_valid_loss = current_loss
                    early_stopping_rounds = 0
                    checkpoint_path = os.path.join(
                        FLAGS.train_dir, 'best_model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

                elif early_stopping_rounds > 5:
                    print("Valition loss didn't improve for 5 rounds... "
                        "Stopping early.")
                    sys.stdout.flush()
                    break

                else:
                    early_stopping_rounds += 1


def main(argv=None):

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
