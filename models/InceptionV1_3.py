import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, Dense
from tensorflow.optimizers import 
from .utils.blocks import inceptionV1_module

def _build_stemV1(model, channels=64):
    stem = Conv2D(channels, (7, 7), activation='relu')(model)
    stem = BatchNormalization()(model)
    stem = MaxPool2D((2, 2))(model)
    return stem

def build_model(input_shape=(256, 256, 4)):
    input_layer = Input(shape=input_shape)

    model = _build_stemV1(input_layer)
    for i in range(3):
        model = inceptionV1_module(model)

    model = GlobalAveragePooling2D()(model)
    model = Dense(28, activation='sigmoid')(model)

    optimizer = tf.train.AdamOptimizer(lr=0.001)
    return model
    
