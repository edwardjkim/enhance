"""
Modified from
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
NUM_CHANNELS = 3

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 233701
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 18119


def read_frames(filename_queue):
  
    class ImageRecord(object):
      pass

    result = ImageRecord()
  
    result.height = IMAGE_HEIGHT
    result.width = IMAGE_WIDTH
    result.depth = NUM_CHANNELS
    image_bytes = result.height * result.width * result.depth

    reader = tf.TFRecordReader()

    _, example_serialized = reader.read(filename_queue)
    features = {
        'prev_frame': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'curr_frame': tf.FixedLenFeature([], dtype=tf.string, default_value='')}
    features = tf.parse_single_example(example_serialized, features)

    prev_image = tf.image.decode_jpeg(features['prev_frame'])
    curr_image = tf.image.decode_jpeg(features['curr_frame'])

    result.image = tf.stack([prev_image, curr_image])

    result.image.set_shape([2, result.height, result.width, result.depth])

    return result


def _generate_image_batch(image, batch_size, shuffle):
    """Construct a queued batch of images.
    Args:
      image: 3-D Tensor of [height, width, 3] of type.float32.
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
      batch_size: Number of images per batch.
      shuffle: boolean indicating whether to use a shuffling queue.
    Returns:
      images: Images. 4D tensor of [batch_size, height, width, 3] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images = tf.train.shuffle_batch(
            [image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=200,
            min_after_dequeue=100)
    else:
        images = tf.train.batch(
            [image],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=200)

    return images


def inputs(filenames, batch_size, shuffle=True):
    """
    """

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
  
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)
  
    # Read examples from files in the filename queue.
    read_input = read_frames(filename_queue)
    reshaped_image = tf.cast(read_input.image, tf.float32)
    reshaped_image = reshaped_image / 127.5 - 1.0

    height = IMAGE_HEIGHT
    width = IMAGE_WIDTH
    channels = NUM_CHANNELS

    # Generate a batch of images by building up a queue of examples.
    return _generate_image_batch(reshaped_image, batch_size, shuffle=shuffle)
