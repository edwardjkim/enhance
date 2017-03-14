"""Converts image data to TFRecords file format with Example protos.

Modified from
https://github.com/tensorflow/models/blob/master/inception/inception/data/build_image_data.py

The image data set is expected to reside in PNG files located in the
following directory structure.

  data_dir/video_id/image0.png
  data_dir/video_id/image1.png
  ...
  data_dir/video_id/weird-image.png
  data_dir/video_id/my-image.png
  ...

where the sub-directory is the unique label associated with these images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of TFRecord files

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-00127-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128

where we have selected 1024 and 128 shards for each data set. Each record
within the TFRecord file is a serialized Example proto. The Example proto
contains the following fields:

  image/image_raw: string containing PNG encoded image in RGB colorspace

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading


import numpy as np
import tensorflow as tf

tf.app.flags.DEFINE_string('train_directory', '/notebooks/shared/videos/youtube/frames',
                           'Training data directory')
tf.app.flags.DEFINE_string('validation_directory', '/tmp',
                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/notebooks/shared/videos/youtube/tfrecords',
                           'Output data directory')

tf.app.flags.DEFINE_integer('train_shards', 128,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 128,
                            'Number of shards in validation TFRecord files.')

tf.app.flags.DEFINE_integer('num_threads', 8,
                            'Number of threads to preprocess the images.')


FLAGS = tf.app.flags.FLAGS


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer):
  """Build an Example proto for an example.

  Args:
    image_buffer: string, PNG encoding of RGB image
  Returns:
    Example proto
  """

  example = tf.train.Example(features=tf.train.Features(feature={
      'image_raw': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._png_data, channels=3)

  def decode_png(self, image_data):
    image = self._sess.run(self._decode_png,
                           feed_dict={self._png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.PNG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, PNG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'r') as f:
    image_data = f.read()

  # Decode the RGB PNG>
  image = coder.decode_png(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  assert height == 360
  width = image.shape[1]
  assert width == 640
  assert image.shape[2] == 3

  return image_data, height, width


def _process_image_files_batch(coder, thread_index, ranges, name, filenames, num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]

      image_buffer, height, width = _process_image(filename, coder)

      example = _convert_to_example(image_buffer)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()


def _process_image_files(name, filenames, num_shards):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    texts: list of strings; each string is human readable, e.g. 'dog'
    labels: list of integer; each integer identifies the ground truth
    num_shards: integer number of shards for this data set.
  """
  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i+1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, filenames, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir):
  """Build a list of all images files and labels in the data set.

  Args:
    data_dir: string, path to the root directory of images.

      Assumes that the image data set resides in PNG files located in
      the following directory structure.

        data_dir/another-image.PNG
        data_dir/my-image.PNG

  Returns:
    filenames: list of strings; each string is a path to an image file.
  """
  print('Determining list of input files and labels from %s.' % data_dir)

  filenames = []

  png_file_path = '{}/*/*.png'.format(data_dir)

  matching_files = tf.gfile.Glob(png_file_path)

  filenames.extend(matching_files)

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]

  print('Found %d PNG files inside %s.' % (len(filenames), data_dir))
  sys.stdout.flush()

  return filenames


def _process_dataset(name, directory, num_shards):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    directory: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    labels_file: string, path to the labels file.
  """
  filenames = _find_image_files(directory)
  _process_image_files(name, filenames, num_shards)


def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  if tf.gfile.Exists(FLAGS.output_directory):
    tf.gfile.DeleteRecursively(FLAGS.output_directory)
  tf.gfile.MakeDirs(FLAGS.output_directory)

  # Run it!
  #_process_dataset('validation', FLAGS.validation_directory, FLAGS.validation_shards)
  _process_dataset('train', FLAGS.train_directory, FLAGS.train_shards)


if __name__ == '__main__':
  tf.app.run()
