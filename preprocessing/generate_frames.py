import os
import subprocess as sp

import numpy as np
import tensorflow as tf
from scipy.misc import imread

from pytube import YouTube
from tqdm import tqdm
from glob import glob


VIDEO_DIR = '/notebooks/shared/videos/webcam'


def get_video_ids_from_tfrecord(filename, match_label=None):

    record_iterator = tf.python_io.tf_record_iterator(path=filename)

    video_ids = []

    print('Reading %s' % filename)

    for string_record in record_iterator:
        
        example = tf.train.Example()
        example.ParseFromString(string_record)

        if match_label:
            labels = example.features.feature["labels"].int64_list.value
            if match_label not in labels:
                    continue
            
        video_ids.append(example.features.feature["video_id"].bytes_list.value[0])

    return video_ids


def mkdir_if_not_exists(dirname):

    print('Creating directory %s' % dirname)

    if not os.path.exists(dirname):
        os.mkdir(dirname)


def download_youtube_video(video_id, resolution='720p'):

    print('Downloading video %s' % video_id)

    try:
        yt = YouTube("http://www.youtube.com/watch?v={}".format(video_id))
    except:
        return False
    
    yt.set_filename("{}".format(video_id))
    yt.filter(resolution=resolution)
    
    try:
        video = yt.get('mp4', resolution)
    except:
        return False

    try:
        video.download(VIDEO_DIR)
    except:
        return False

    return True


def generate_png(filename, output_dir, fps=2, vertical=360):

    command = [
        'ffmpeg',
        '-i', filename,
        '-vf', 'fps=%d' % fps,
        '-vf', 'scale=-1:%d' % vertical,
        os.path.join(output_dir, 'out%6d.png')
    ]
    
    pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)

    return


def main():

    mkdir_if_not_exists(VIDEO_DIR)
    mkdir_if_not_exists(os.path.join(VIDEO_DIR, 'frames'))

    with open('webcam.txt') as f:
        video_ids = [line.strip() for line in f]

    for v in video_ids:

        video_path = os.path.join(VIDEO_DIR, "{}.mp4".format(v))

        if not os.path.exists(video_path):
            success = download_youtube_video(v, resolution='720p')

        if success:
            output_dir = os.path.join(VIDEO_DIR, 'frames', v)

            if not os.path.exists(output_dir):
                mkdir_if_not_exists(output_dir)
                generate_png(os.path.join(VIDEO_DIR, '%s.mp4' % v), output_dir, fps=2, vertical=240)


if __name__ == '__main__':
    main()
