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


IMAGE_HEIGHT = sres.IMAGE_HEIGHT
IMAGE_WIDTH = sres.IMAGE_WIDTH
NUM_CHANNELS = sres.NUM_CHANNELS


def split_and_resize(images):

    images = tf.split(images, 2, axis=1)

    downsampled_images = []
    for i in range(2):
        downsampled_height = int(IMAGE_HEIGHT / FLAGS.upscale_factor)
        downsampled_width = int(IMAGE_WIDTH / FLAGS.upscale_factor)
        image = tf.squeeze(images[i], squeeze_dims=[1])
        downsampled_images.append(tf.image.resize_images(
            image,
            [downsampled_height, downsampled_width],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
    downsampled_images = tf.stack(downsampled_images)
    downsampled_images = tf.transpose(
        downsampled_images, perm=[1, 0, 2, 3, 4])

    return downsampled_images


def train():

    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
    
        # Get images
        real_images = sres.inputs()
        downsampled_real_images = split_and_resize(real_images)
    
        # Build a Graph that computes the logits predictions from the inference
        # model.
        fake_images = sres.generator(downsampled_real_images)
        downsampled_fake_images = split_and_resize(fake_images)
    
        # Calculate loss.
        gen_loss = sres.loss(real_images, fake_images)
    
        # Validation data
        valid_real_images = sres.inputs(eval_data=True)
        downsampled_valid_real_images = split_and_resize(valid_real_images)
        valid_fake_images = sres.generator(downsampled_valid_real_images)
    
        valid_loss = sres.loss(valid_real_images, valid_fake_images)
    
        # Build a Graph that trains the model with one batch of examples and
        # updates the model parameters.
        train_op = sres.train(gen_loss)
    
        # Build an initialization operation to run below.
        init_op = tf.global_variables_initializer()
    
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())
    
        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init_op)
    
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)
    
        best_valid_loss = np.inf
        early_stopping_rounds = 0
    
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            _, gen_loss_value = sess.run([train_op, gen_loss])
            duration = time.time() - start_time
      
            assert not np.isnan(gen_loss_value), 'Model diverged with NaN loss'
      
            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
        
                format_str = ('%s: step %d, gen loss = %.6f'
                    '(%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (
                  datetime.now(), step, gen_loss_value,
                  examples_per_sec, sec_per_batch))
                sys.stdout.flush()
        
            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
        
                current_loss = 0
                num_steps_per_eval = int(
                    sres.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / FLAGS.batch_size)
                for _ in range(num_steps_per_eval):
                  current_loss += sess.run(valid_loss)
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
                    checkpoint_path = os.path.join(
                        FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    print("Valition loss didn't improve for 5 rounds... "
                        "Stopping early.".format())
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
