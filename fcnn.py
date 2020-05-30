from docutils.nodes import Sequential
from keras.layers.core import Dense, Dropout
import keras


def model():
    model = Sequential([
        Dense(
            180, activation='relu', input_shape=(254,)),
        Dropout(0.2),
        Dense(180, activation='relu'),
        Dropout(0.2),
        Dense(180, activation='relu'),
        Dropout(0.2),
        Dense(180, activation='relu'),
        Dropout(0.2),
        Dense(180, activation='relu'),
        Dropout(0.2),
        Dense(180, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[
                  'accuracy', keras.metrics.AUC(), keras.metrics.Precision(), keras.metrics.Recall()])

    return model
