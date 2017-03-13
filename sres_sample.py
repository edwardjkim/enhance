from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from scipy import misc

import sres


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/sres_eval',
    """Directory where to write event logs.""")
tf.app.flags.DEFINE_string(
    'checkpoint_dir', '/tmp/sres',
    """Directory where to read model checkpoints.""")


def read_npy(filenames):
    """
    Goes through a list of file paths and returns a stacked numpy array.

    Parameters
    ----------
    filenames: A list of strings.

    Returns
    -------
    A numpy array of shape (num_examples, rows, cols, depth).
    """
    images = list()
  
    for f in tqdm(filenames):
  
        image = misc.imread(f)

        if image.shape[0] != 360:
            continue
        if image.shape[1] != 640:
            continue
        if image.shape[2] != 3:
            continue

        images.append(image)
  
    images = np.stack(images)

    return images


def sample():
 
    filenames = glob.glob('/notebooks/shared/videos/youtube/frames/*/*.png')
    sample_images = read_npy(filenames[-500:])

    print(sample_images.min(), sample_images.max())

    downsized_images = np.zeros([64, 180, 320, 3])

    for i in tqdm(range(64)):
        resized = misc.imresize(sample_images[i], [180, 320], interp='nearest')
        downsized_images[i] = (resized / 127.5) - 1.0

    print(downsized_images.min(), downsized_images.max())
    
    # Get images
    real_images = tf.placeholder(
        tf.float32,
        shape=[64, 180, 320, 3])

    fake_images = sres.generator(real_images)

    saver = tf.train.Saver()
  
    with tf.Session() as sess:
  
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
  
      if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Checkpoint restored from {}'.format(ckpt.model_checkpoint_path))
      else:
        print('No checkpoint file found')
        return

      generated_images = sess.run(fake_images, feed_dict={real_images: downsized_images})
 
      np.save(os.path.join(FLAGS.eval_dir, "input.npy"), downsized_images)
      np.save(os.path.join(FLAGS.eval_dir, "samples.npy"), generated_images)
  
    return


def main(argv=None):  # pylint: disable=unused-argument

  if tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.DeleteRecursively(FLAGS.eval_dir)
  tf.gfile.MakeDirs(FLAGS.eval_dir)

  sample()


if __name__ == '__main__':
  tf.app.run()
