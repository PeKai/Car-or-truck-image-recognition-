import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, TensorBoard
from tensorflow.keras.preprocessing import image_dataset_from_directory
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from tensorflow.keras.layers.experimental import preprocessing
import datetime
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    '/Users/maximilian/Downloads/archive (3)/train',
    target_size=(128, 128),
    batch_size=20,
    class_mode='binary',
    shuffle=True,
)

test_generator = test_datagen.flow_from_directory(
    '/Users/maximilian/Downloads/archive (3)/valid',
    target_size=(128, 128),
    batch_size=20,
    class_mode='binary',
    shuffle=False,
)

model = Sequential([

    preprocessing.RandomContrast(0.3),

    layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same',
                  input_shape=[128, 128, 3]),
    layers.MaxPool2D(),

    layers.Flatten(),
    layers.Dense(units=12, activation="relu"),
    layers.Dropout(0.4),
    layers.BatchNormalization(),
    layers.Dense(units=12, activation="relu"),

    layers.Dense(units=1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

model.fit(train_generator,
          validation_data=test_generator,
          epochs=50,
          callbacks=[EarlyStopping(monitor='loss', restore_best_weights=True, patience=5), TerminateOnNaN()])

model.summary()
