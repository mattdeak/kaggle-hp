from abc import ABC, abstractmethod
import tensorflow as tf

class PreprocessingFunction(ABC):

    @abstractmethod
    def __call__(self, X, y):
        """The preprocessing function to be applied to a tf.Dataset"""

class OneHotLabels(PreprocessingFunction):

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, image, labels):
        ohe_labels = tf.reduce_sum(
            tf.one_hot(labels, self.n_classes
            ), axis=0)
        return image, ohe_labels

class ResizeImage(PreprocessingFunction):

    def __init__(self, size, method=tf.image.ResizeMethod.BILINEAR):
        self.size = size
        self.method = method

    def __call__(self, image, labels):
        """Resizes an image"""
        resized = tf.image.resize_images(
            image, 
            (self.size), 
            method=self.method)
        return resized, labels

class FloatifyImage(PreprocessingFunction):

    def __call__(self, image, labels):
        """Converts an image from 0-255 int to 0-1 float format."""
        return image / 255, labels

class DropChannel(PreprocessingFunction):

    CHANNELS = ['red', 'green', 'blue', 'yellow']
    def __init__(self, channel):
        assert channel in self.CHANNELS, f"Channel {channel} not supported. Must be one of {self.CHANNELS}"
        
        self.channel = self.CHANNELS.index(channel)

    def __call__(self, image, labels):
        """Drops a color channel from an image"""
        #TODO:Implement

class RandomAugmentation(PreprocessingFunction):

    def __init__(self, vertical_flip=True, horizontal_flip=True, contrast_deltas=None, brightness_maxdelta=None):
        self.vertical = vertical_flip
        self.horizontal = horizontal_flip
        self.contrast_deltas = contrast_deltas
        self.brightness_delta = brightness_maxdelta

    def __call__(self, image, labels):
        if self.vertical:
            image = tf.image.random_flip_up_down(image)
        
        if self.horizontal:
            image = tf.image.random_flip_left_right(image)

        if self.brightness_delta:
            image = tf.image.random_brightness(image, self.brightness_delta)
            image = tf.clip_by_value(image, 0.0, 1.0) # In case brightness scaling goes too high or low

        if self.contrast_deltas:
            image = tf.image.random_contrast(image, self.contrast_deltas[0], self.contrast_deltas[1])
            image = tf.clip_by_value(image, 0.0, 1.0) 

        return image, labels

