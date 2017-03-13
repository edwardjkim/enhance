from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf


IMAGE_ROWS = 360
IMAGE_COLS = 640
NUM_CHANNELS = 3

# Global constants describing the data set.
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2262
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 96


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

    reader = tf.FixedLengthRecordReader(record_bytes=image_bytes)
    result.key, value = reader.read(filename_queue)
  
    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)
  
    result.uint8image = tf.reshape(
        tf.strided_slice(record_bytes, [0], [image_bytes]),
        [result.height, result.width, result.depth])
  
    return result


def _generate_image_batch(image, min_queue_examples, batch_size, shuffle):
    """
    """
    num_preprocess_threads = 16
    if shuffle:
      images = tf.train.shuffle_batch(
          [image],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size,
          min_after_dequeue=min_queue_examples)
    else:
      images = tf.train.batch(
          [image],
          batch_size=batch_size,
          num_threads=num_preprocess_threads,
          capacity=min_queue_examples + 3 * batch_size)
  
    return images


def distorted_inputs(data_dir, batch_size):
    """
    """
    filename = '/notebooks/shared/videos/youtube/tfrecords/train.tfrecords'
  
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

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    #distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
  
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(reshaped_image)
  
    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    #distorted_image = tf.image.random_brightness(
    #    distorted_image, max_delta=63)
    #distorted_image = tf.image.random_contrast(
    #    distorted_image, lower=0.2, upper=1.8)
  
    # Subtract off the mean and divide by the variance of the pixels.
    #float_image = tf.image.per_image_standardization(distorted_image)
  
    # Set the shapes of tensors.
    distorted_image.set_shape([height, width, channels])
  
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print('Filling queue with %d images before starting to train. '
        'This will take a few minutes.' % min_queue_examples)
  
    # Generate a batch of images by building up a queue of examples.
    return _generate_image_batch(distorted_image, min_queue_examples, batch_size, shuffle=True)


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
