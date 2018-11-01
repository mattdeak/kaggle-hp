import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, Dense, Flatten
from .utils.blocks import inceptionV1_module
from .utils.metrics import f1_macro, make_class_specific_f1

IMAGE_SHAPE = (256, 256)

def _build_stemV1(model, channels=64):

    stem = Conv2D(channels, (7, 7), activation='relu', kernel_regularizer=l2(1e-4))(model)
    stem = BatchNormalization()(stem)
    stem = MaxPool2D((2, 2))(stem)
    return stem

def build_model(input_shape=(256, 256, 4), per_class_f1=True)):
    input_layer = Input(shape=input_shape)

    model = _build_stemV1(input_layer)
    for i in range(3):
        model = inceptionV1_module(model, regularizer=l2(1e-4))

    model = GlobalAveragePooling2D()(model)
    model = Flatten()(model)
    outputs = Dense(28, activation='sigmoid', kernel_regularizer=l2(1e-4))(model)

    optimizer = tf.train.AdamOptimizer(0.0001)

    model = keras.Model(inputs=input_layer, outputs=outputs)

    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=[f1_macro, 'accuracy'])

    return model
    
