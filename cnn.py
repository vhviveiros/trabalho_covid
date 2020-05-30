from keras.engine.sequential import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Flatten
import keras


def model():
    model = Sequential()
    model.add(
        Conv2D(32, (3, 3), input_shape=(512, 512, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
        'accuracy', keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])

    return model
