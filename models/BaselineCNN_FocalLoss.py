import tensorflow.keras as keras
import tensorflow as tf

from .utils.metrics import f1_macro, make_class_specific_f1
from .utils.losses import focal_loss

from lib.preprocessing import OneHotLabels, ResizeImage, FloatifyImage
from lib.utils import load_train, load_validation
from lib.config import NUM_CLASSES

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, Flatten
from tensorflow.keras.layers import MaxPooling2D, Dropout


def build_model(input_shape=(264, 264, 4), per_class_f1=True)):
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

    optimizer = tf.train.SGD(0.1, 0.8)
    focal = focal_loss()

    metrics = [f1_macro]

    if per_class_f1:
        all_metrics = [make_class_specific_f1(i) for i in range(NUM_CLASSES)]
        metrics = metrics + all_metrics

    model.compile(optimizer=optimizer,
                loss=focal,
                metrics=metrics)
    
    return model