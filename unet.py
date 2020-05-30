from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.merge import concatenate
from keras.engine.training import Model
from keras.engine.input_layer import Input
import keras


def model(input_size=(512, 512, 3)):
    inputs = keras.layers.Input(input_size)
    s = keras.layers.Lambda(lambda x: x / 255)(inputs)

    c1 = keras.layers.Conv2D(16, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(s)
    c1 = keras.layers.Dropout(0.1)(c1)
    c1 = keras.layers.Conv2D(16, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(c1)
    p1 = keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = keras.layers.Conv2D(32, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(p1)
    c2 = keras.layers.Dropout(0.1)(c2)
    c2 = keras.layers.Conv2D(32, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(c2)
    p2 = keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = keras.layers.Conv2D(64, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(p2)
    c3 = keras.layers.Dropout(0.2)(c3)
    c3 = keras.layers.Conv2D(64, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(c3)
    p3 = keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = keras.layers.Conv2D(128, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(p3)
    c4 = keras.layers.Dropout(0.2)(c4)
    c4 = keras.layers.Conv2D(128, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(c4)
    p4 = keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = keras.layers.Conv2D(256, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(p4)
    c5 = keras.layers.Dropout(0.3)(c5)
    c5 = keras.layers.Conv2D(256, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(c5)

    u6 = keras.layers.Conv2DTranspose(
        128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = keras.layers.concatenate([u6, c4])
    c6 = keras.layers.Conv2D(128, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(u6)
    c6 = keras.layers.Dropout(0.2)(c6)
    c6 = keras.layers.Conv2D(128, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(c6)

    u7 = keras.layers.Conv2DTranspose(
        64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = keras.layers.concatenate([u7, c3])
    c7 = keras.layers.Conv2D(64, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(u7)
    c7 = keras.layers.Dropout(0.2)(c7)
    c7 = keras.layers.Conv2D(64, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(c7)

    u8 = keras.layers.Conv2DTranspose(
        32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = keras.layers.concatenate([u8, c2])
    c8 = keras.layers.Conv2D(32, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(u8)
    c8 = keras.layers.Dropout(0.1)(c8)
    c8 = keras.layers.Conv2D(32, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(c8)

    u9 = keras.layers.Conv2DTranspose(
        16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = keras.layers.concatenate([u9, c1], axis=3)
    c9 = keras.layers.Conv2D(16, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(u9)
    c9 = keras.layers.Dropout(0.1)(c9)
    c9 = keras.layers.Conv2D(16, (3, 3), activation=keras.activations.elu, kernel_initializer='he_normal',
                             padding='same')(c9)

    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = keras.Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
        'accuracy', keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])

    return model
