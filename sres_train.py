from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
from datetime import datetime
import time
import numpy as np
import tensorflow as tf

import sres


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/sres',
    """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer(
    'max_steps', 100000,
    """Number of batches to run.""")
tf.app.flags.DEFINE_boolean(
    'log_device_placement', False,
    """Whether to log device placement.""")


def train():

  with tf.Graph().as_default():

    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images
    real_images = sres.distorted_inputs()

    downsized_images = tf.image.resize_images(
        real_images, [int(360 // FLAGS.upscale_factor), int(640 // FLAGS.upscale_factor)], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    # Build a Graph that computes the logits predictions from the inference model.
    fake_images = sres.generator(downsized_images)

    # Calculate loss.
    loss = sres.loss(real_images, fake_images)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = sres.train(loss)

    # Build an initialization operation to run below.
    init_op = tf.group(
      tf.global_variables_initializer(),
      tf.local_variables_initializer())

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build an initialization operation to run below.
    init_op = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
      log_device_placement=FLAGS.log_device_placement))
    sess.run(init_op)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss_lab = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.6f, '
                      '(%.1f examples/sec; %.3f sec/batch)\n')
        sys.stdout.write(format_str % (
          datetime.now(), step, loss_value, examples_per_sec, sec_per_batch
        ))
        sys.stdout.flush()

      # Save the model checkpoint periodically.
      if step % 100 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):

    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
  
    train()


if __name__ == '__main__':

    tf.app.run()
