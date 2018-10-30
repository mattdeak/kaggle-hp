import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, Dense
from .utils.blocks import inceptionV1_module
from .utils.metrics import f1_macro

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
    outputs = Dense(28, activation='sigmoid')(model)

    optimizer = tf.train.AdamOptimizer(0.001)

    model = keras.Model(inputs=input_layer, outputs=outputs)

    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[f1_macro, 'accuracy'])

    return model
    
