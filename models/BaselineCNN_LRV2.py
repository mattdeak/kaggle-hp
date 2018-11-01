import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, Dense, Flatten
from .utils.blocks import inceptionV1_module
from .utils.metrics import f1_macro, make_class_specific_f1

def build_model(input_shape=(264, 264, 4), per_class_f1=True):
    """Builds the testModel1 Architecture.
    """
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation=tf.nn.relu)(inputs)
    x = MaxPooling2D((2, 2))(x)

    for i in range(2):
        x = Conv2D(64, (3, 3), activation=tf.nn.relu)(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(264, activation=tf.nn.relu)(x)
    x = Dropout(.5)(x)
    x = Dense(264, activation=tf.nn.relu)(x)
    x = Dropout(.5)(x)
    
    outputs = Dense(28, activation=tf.nn.sigmoid)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.train.AdamOptimizer(0.0001)

    metrics = [f1_macro]

    if per_class_f1:
        all_metrics = [make_class_specific_f1(i) for i in range(NUM_CLASSES)]
        metrics = metrics + all_metrics

    model.compile(optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=metrics)
    
    return model
    
