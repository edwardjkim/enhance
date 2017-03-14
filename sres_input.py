from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


IMAGE_ROWS = 360
IMAGE_COLS = 640
NUM_CHANNELS = 3


def read_frames(filename_queue):
    """
    """
  
    class ImageRecord(object):
      pass

    result = ImageRecord()
  
    result.height = IMAGE_ROWS
    result.width = IMAGE_COLS
    result.depth = NUM_CHANNELS
    image_bytes = result.height * result.width * result.depth

    reader = tf.TFRecordReader()

    _, example_serialized = reader.read(filename_queue)
    features = {
        'image_raw': tf.FixedLenFeature([], dtype=tf.string, default_value='')}
    features = tf.parse_single_example(example_serialized, features)
    result.image = tf.image.decode_png(features['image_raw'])
    result.image.set_shape([result.height, result.width, result.depth])
  
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


def distorted_inputs(filenames, batch_size):
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

    height = IMAGE_ROWS
    width = IMAGE_COLS
    channels = NUM_CHANNELS

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(reshaped_image)

    # Subtract off the mean and divide by the variance of the pixels.
  
    # Set the shapes of tensors.
    #distorted_image.set_shape([height, width, channels])

    # Generate a batch of images by building up a queue of examples.
    return _generate_image_batch(distorted_image, batch_size, shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """
    """
    if not eval_data:
        filename = '/notebooks/shared/videos/youtube/tfrecords/train.tfrecords'
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        print(filename)
    else:
        filename = '/notebooks/shared/videos/youtube/tfrecords/test.tfrecords'
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        print(filename)
  
    if not tf.gfile.Exists(filename):
        raise ValueError('Failed to find file: ' + filename)
  
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer([filename])
  
    # Read examples from files in the filename queue.
    read_input = read_frames(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
  
    height = IMAGE_ROWS
    width = IMAGE_COLS
    channels = NUM_CHANNELS

    # Subtract off the mean and divide by the variance of the pixels.
    #float_image = tf.image.per_image_standardization(reshaped_image)
  
    # Set the shapes of tensors.
    reshaped_image.set_shape([height, width, channels])
  
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
  
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_batch(reshaped_image, min_queue_examples, batch_size, shuffle=False)
