from .config import *
import tensorflow as tf
from sklearn.metrics import f1_score

def _decode_img(img, height=512, width=512):
    image = tf.image.decode_image(img, dtype=tf.float32)
    image = tf.reshape(image, (512, 512))
    
    return image

def _parse_function(example_proto):
    """Parses TF records
    
    Arguments:
        example_proto {tf.Example} -- [description]
    
    Returns:
        list -- A list in the form of [Id, Image, Labels]
    """
    features = {
        'id' : tf.VarLenFeature(tf.string),
        'height' : tf.FixedLenFeature([], tf.int64),
        'width' : tf.FixedLenFeature([], tf.int64),
        'image_green' : tf.FixedLenFeature([], tf.string),
        'image_red' : tf.FixedLenFeature([], tf.string),
        'image_blue' : tf.FixedLenFeature([], tf.string),
        'image_yellow' : tf.FixedLenFeature([], tf.string),
        'labels' : tf.VarLenFeature(tf.int64)
    }

    
    parsed = tf.parse_single_example(example_proto, features)
    
    # Reconstruct the Images
    channels = ['red', 'green', 'blue', 'yellow']
    h = features['height']
    w = features['width']
    
    images = tf.stack([_decode_img(
                parsed[f'image_{color}'], height=h, width=w) 
            for color in channels], axis=2)
    
    # Convert labels to dense tensor (necessary?)
    labels = parsed['labels']
    labels = tf.sparse_to_dense(labels.indices, labels.dense_shape, labels.values)
    
    return images, labels

def _load_dataset(tfrecord_path, **kwargs):
    dataset = tf.data.TFRecordDataset(tfrecord_path, **kwargs)
    dataset = dataset.map(_parse_function)
    return dataset

def load_train(**kwargs):
    path = f'{KAGGLE_TRAIN}/tfrecords/train.tfrecords'
    return _load_dataset(path, **kwargs)

def load_validation(**kwargs):
    path = f'{KAGGLE_TRAIN}/tfrecords/val.tfrecords'
    return _load_dataset(path, **kwargs)

def load_test(**kwargs):
    path = f'{KAGGLE_TRAIN}/tfrecords/test.tfrecords'
    return _load_dataset(path, **kwargs)

#def build_and_train_estimator(keras_model, input_fn, val_input_fn=None, )


