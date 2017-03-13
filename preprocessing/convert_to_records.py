"""
Modified from
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import glob

from tqdm import tqdm

import numpy as np
from scipy import misc
import tensorflow as tf


IMAGE_ROWS = 360
IMAGE_COLS = 640
NUM_CHANNELS = 3

FLAGS = None


def _bytes_feature(value):

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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

        if image.shape[0] != IMAGE_ROWS:
            continue
        if image.shape[1] != IMAGE_COLS:
            continue
        if image.shape[2] != NUM_CHANNELS:
            continue

        image = (image / 127.5) - 1.0

        images.append(image)
  
    images = np.stack(images)

    return images


def convert_to(images, name):
    """
    Converts an image dataset to tfrecords.

    Paramters
    ---------
    images: A numpy array of shape (num_examples, rows, cols, depth).
    name: Output filename (withtout the extension).
          Specify the directory with FLAGS.directory.

    Returns
    -------
    None
    """

    num_examples = images.shape[0]
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]
  
    if not os.path.exists(FLAGS.directory):
        os.mkdir(FLAGS.directory)
  
    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def main(unused_argv):

    filenames = glob.glob('/notebooks/shared/videos/youtube/frames/*/*.{}'.format(FLAGS.type))

    train_images = read_npy(filenames[:5000])
    test_images = read_npy(filenames[-500:])

    print(train_images.shape, train_images.min(), train_images.max())
    print(test_images.shape, test_images.min(), test_images.max())

    # Convert to Examples and write the result to TFRecords.
    convert_to(train_images, 'train')
    convert_to(test_images, 'test')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='/notebooks/shared/videos/youtube/tfrecords',
        help='Directory to download data files and write the converted result'
    )
    parser.add_argument(
        '--type',
        type=str,
        default='png',
        help='Image type. We use this string for file extension.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
