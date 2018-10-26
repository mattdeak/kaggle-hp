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

