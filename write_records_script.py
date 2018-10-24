import sys

import tensorflow as tf 
import matplotlib.image as im
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from config import *

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    if isinstance(value, list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    # Else..
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def load_image(image_id, directory=KAGGLE_TRAIN, channels=['green', 'blue', 'red', 'yellow']):
    """Loads image channels from an image ID
    
    Arguments:
        image_id {str} -- The ID of the image
    
    Keyword Arguments:
        channels {list} -- A list of colour channels to include (default: {['green', 'blue', 'red', 'yellow']})
    
    Returns:
        dict -- A dictionary in the form of {channel : image_info}
    """
    images = {channel : im.imread(f'{directory}/{image_id}_{channel}.png') for channel in channels}

    return images

def extract_id(filename):
    """Extracts the image ID from the file name
    
    Arguments:
        filename {str} -- The full file path
    
    Returns:
        str -- The ID of the file
    """
    return filename.split('_')[0]

def create_record(out_filename, data):
    """Creates a TFRecord dataset
    
    Arguments:
        out_filename {str} -- The path to the output file
        id_info {[type]} -- [description]
    """
    ids, labels = zip(*data)

    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in tqdm(range(len(ids))):
        img_id = ids[i]
        images = load_image(img_id)
        label = labels[i]

        feature = {
            'id' : _bytes_feature(bytes(img_id, 'utf-8')),
            'height' : _int64_feature(images['green'].shape[0]),
            'width' : _int64_feature(images['green'].shape[1]),
            'image_green' : _bytes_feature(images['green'].tostring()),
            'image_red' : _bytes_feature(images['red'].tostring()),
            'image_blue' : _bytes_feature(images['blue'].tostring()),
            'image_yellow' : _bytes_feature(images['yellow'].tostring()),
            'labels' : _int64_feature(label)
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()


def main():
    os.chdir('..')
    # Collect All Unique Ids and Labels and
    data = pd.read_csv(f'{KAGGLE_TRAIN}/train.csv')
    data.Target = data.Target.apply(lambda x: [int(a) for a in x.split(' ')])
    data = list(zip(data.Id, data.Target))
    # Randomize and divide into 70% train, 15% validation and 15% test
    np.random.shuffle(data)
    
    N = len(data)
    train_data = data[:int(0.7*N)]
    val_data = data[int(0.7*N):int(0.85*N)]
    test_data = data[int(0.85*N):]

    print("Writing Train Records")
    create_record(f'{KAGGLE_TRAIN}/tfrecords/train.tfrecords', train_data)

    print("Writing Validation Records")
    create_record(f'{KAGGLE_TRAIN}/tfrecords/val.tfrecords', val_data)

    print("Writing Test Records")
    create_record(f'{KAGGLE_TRAIN}/tfrecords/test.tfrecords', test_data)

if __name__ == '__main__':
    main()
    
