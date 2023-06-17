from google.colab import drive
drive.mount('/content/drive')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import layers
from keras.models import Sequential

TRAIN_PATH = '/content/drive/MyDrive/DataSets/Classification_data/train'
TEST_PATH = '/content/drive/MyDrive/DataSets/Classification_data/test'
batch_size = 32
img_height = 150
img_width = 150

def main():
  train_ds = create_data(TRAIN_PATH)
  val_ds = create_data(TEST_PATH)
  class_names = train_ds.class_names
  num_classes = len(class_names)
  train_ds, val_ds = preprocess_data(train_ds, val_ds)
  normalization_layer = layers.Rescaling(1./255)
  callback = get_callback()
  model = get_model(num_classes)
  model = compile_model(model)
  history = train_model(model, 1000, train_ds, val_ds, callback)
  return history

def create_data(PATH):
  train_ds = tf.keras.utils.image_dataset_from_directory(
  PATH,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  return train_ds

def preprocess_data(train_ds, val_ds):
  # Use buffered prefetching, so you can yield data from disk without having I/O become blocking.
  AUTOTUNE = tf.data.AUTOTUNE
  train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
  return train_ds, val_ds

def get_callback():
  # Initialize early stopping to prevent overfitting
  callback = keras.callbacks.EarlyStopping(
      monitor='val_loss',
      min_delta=0,
      patience=10,
      verbose=0,
      mode='auto',
      baseline=None,
      restore_best_weights=True,
      start_from_epoch=0
  )
  return callback

def get_model(num_classes):
  model = Sequential([
      layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
      layers.Conv2D(64, 3, padding="same", activation="relu"),
      layers.Conv2D(3, 3, padding="same", activation="relu"),
      layers.MaxPooling2D(),
      layers.Conv2D(4, 3, padding="same", activation="relu"),
      layers.MaxPooling2D(),
      layers.Conv2D(9, 3, padding="same", activation="relu"),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation="relu"),
      layers.Dropout(0.2),
      layers.Dense(64, activation="sigmoid"),
      layers.Dense(num_classes)
  ])
  return model

def compile_model(model):
  model.compile(optimizer="adam",
                loss=keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"])
  model.summary()
  return model

def train_model(model, epochs, train_ds, val_ds, callback):
  history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs,
      callbacks=[callback]
  )
  return history

history = main()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(loss))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
