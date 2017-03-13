import os
import subprocess as sp

import numpy as np
import tensorflow as tf

from pytube import YouTube
from tqdm import tqdm


VIDEO_DIR = '/notebooks/shared/videos/youtube/'

tfrecords_filename = '/notebooks/shared/videos/yt8m/train0_.tfrecord'

record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

video_ids = []

for string_record in record_iterator:
    
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    video_ids.append(example.features.feature["video_id"].bytes_list.value[0])

if not os.path.exists(os.path.join(VIDEO_DIR, 'frames')):
    os.mkdir(os.path.join(VIDEO_DIR, 'frames'))

for video_id in tqdm(video_ids):

    try:
        yt = YouTube("http://www.youtube.com/watch?v={}".format(video_id))
    except:
        continue
    
    if os.path.exists(os.path.join(VIDEO_DIR, "{}.mp4".format(video_id))):
        os.remove(os.path.join(VIDEO_DIR, "{}.mp4".format(video_id)))
    yt.set_filename("{}".format(video_id))
    yt.filter(resolution='360p')
    
    try:
        video = yt.get('mp4', '360p')
    except:
        continue
    
    video.download(VIDEO_DIR)

    if not os.path.exists(os.path.join(VIDEO_DIR, 'frames', video_id)):
        os.mkdir(os.path.join(VIDEO_DIR, 'frames', video_id))


    command = [
        'ffmpeg',
        '-i', os.path.join(VIDEO_DIR, "{}.mp4".format(video_id)),
        '-vf', 'fps=1', # 1 frame per second
        os.path.join(VIDEO_DIR, 'frames', '{}'.format(video_id), 'out%6d.png')
    ]
    
    pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
