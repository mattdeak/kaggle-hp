from abc import ABC, abstractmethod
import tensorflow as tf

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

class Maybe(PreprocessingFunction):
    
    def __init__(self, preprocessor, chance=0.5):
        """A preprecessing super-layer that gives another preprocessor a
        predefined chance of actually being executed.
        
        Arguments:
            preprocessor {PreprocessingFunction} -- The preprocessor
        
        Keyword Arguments:
            chance {float} -- The chance of a preprocessor being executed (default: {0.5})
        """
        assert isinstance(preprocessor, PreprocessingFunction), "Preprocessor must be a Preprocessing Function"
        assert chance < 1.0 and chance > 0.0, "Chance should be between 0 and 1 exclusive"
        self.preprocessor = preprocessor
        self.chance = chance
    
        
    def __call__(self, image, labels):
        r = tf.random_uniform([1])[0]
        
        cond = tf.cond(
            r < self.chance,
            true_fn=lambda: self.preprocessor(image, labels),
            false_fn=lambda: (image, labels))
        return cond
        
class RandomCrop(PreprocessingFunction):
    
    def __init__(self, input_shape=(512, 512, 4), min_crop_pct=0.05, max_crop_pct=0.4):
        """Randomly crops an image input. 
        
        Keyword Arguments:
            input_shape {tuple} -- Expected shape of the incoming images (default: {(512, 512, 4)})
            min_crop_pct {float} -- The minimum crop percentage. A larger value means the resulting crops will be smaller in size. (default: {0.05})
            max_crop_pct {float} -- [description] The maximum crop percentage.  (default: {0.4})
        """
        self.input_shape = input_shape
        self.min_crop = min_crop_pct
        self.max_crop = max_crop_pct
        
    def __call__(self, image, labels):
        crop_pct = tf.random_uniform([1], minval=self.min_crop, maxval=self.max_crop)[0]
        cropped_size = tf.cast(self.input_shape[0] * (1 - crop_pct), tf.int32)
        cropped = tf.random_crop(image, (cropped_size, cropped_size, self.input_shape[-1]))
        return cropped, labels

class DropChannel(PreprocessingFunction):

    channel_dict = {
        'red' : 0,
        'green' : 1,
        'blue' : 2,
        'yellow' : 3
    }

    channels = ['red', 'green', 'blue', 'yellow']
    def __init__(self, drop_labels=[]):
        self.keep_labels = [self.channel_dict[ch] for ch in self.channels if ch not in drop_labels]

    def __call__(self, image, labels):
        """Drop the corresponding labels from the tensor"""
        modified = tf.stack([image[:, :, i] for i in self.keep_labels], axis=-1)
        return modified, labels

class DropLabels(PreprocessingFunction):

    def __call__(self, image, labels):
        return image
            

