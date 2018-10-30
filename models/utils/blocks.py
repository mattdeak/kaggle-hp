import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, MaxPool2D

def inceptionV1_module(model, filters=64, regularizer=None):


    tower_1 = Conv2D(filters, (1, 1), padding='same', activation='relu', kernel_regularizer=regularizer)(model)
    tower_1 = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizer)(tower_1)
    
    tower_2 = Conv2D(64, (1,1), padding='same', activation='relu', kernel_regularizer=regularizer)(model)
    tower_2 = Conv2D(64, (5,5), padding='same', activation='relu', kernel_regularizer=regularizer)(tower_2)

    tower_3 = MaxPool2D((3,3), strides=(1,1), padding='same')(model)
    tower_3 = Conv2D(64, (1,1), padding='same', activation='relu', kernel_regularizer=regularizer)(tower_3)


    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)
    return output