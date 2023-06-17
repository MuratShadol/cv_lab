from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential
from keras import layers

TRAIN_PATH = '/content/drive/MyDrive/DataSets/Classification_data/train'
TEST_PATH = '/content/drive/MyDrive/DataSets/Classification_data/test'

batch_size = 32
img_height = 150
img_width = 150

def main():
  train_ds = create_train_set(TRAIN_PATH)
  val_ds = create_val_set(TEST_PATH)
  class_names = train_ds.class_names
  num_classes = len(class_names)
  train_ds, val_ds = preprocess_data(train_ds, val_ds)
  data_augmentation = Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),]
  )
  base_model = get_base_model()
  callback = get_callback()
  model = train_top_layer(base_model, train_ds, val_ds, callback, data_augmentation)
  train_entire_model(base_model, model, train_ds, val_ds, 1000, callback)


def create_train_set(TRAIN_PATH):
  train_ds = tf.keras.utils.image_dataset_from_directory(
  TRAIN_PATH,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  return train_ds

def create_val_set(TEST_PATH):
  val_ds = tf.keras.utils.image_dataset_from_directory(
  TEST_PATH,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
  return val_ds


def preprocess_data(train_ds, val_ds):
  # Use buffered prefetching, so you can yield data from disk without having I/O become blocking.
  AUTOTUNE = tf.data.AUTOTUNE
  train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
  return train_ds, val_ds



def get_base_model():
  base_model = keras.applications.Xception(
      weights='imagenet',  # Load weights pre-trained on ImageNet
      input_shape=(150, 150, 3),
      include_top=False,
      classes=6
  )
  return base_model

# Freeze the base_model
def train_top_layer(base_model, train_ds, val_ds, callback, data_augmentation):
  base_model.trainable = False

  # Create new model on top
  inputs = keras.Input(shape=(150, 150, 3))
  x = data_augmentation(inputs) # apply random data augmentation

  # Rescale the data to [-1; 1] for Xception
  scale_layer = layers.Rescaling(scale=1 / 127.5, offset=-1)
  x = scale_layer(x)

  # The base model contains batchnorm layers. We want to keep them in inference mode
  # when we unfreeze the base model for fine-tuning, so we make sure that the
  # base_model is running in inference mode here
  x = base_model(x, training=False)
  x = keras.layers.GlobalAveragePooling2D()(x)
  x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
  outputs = keras.layers.Dense(6, activation='softmax')(x)
  model = keras.Model(inputs, outputs)
  model.summary()
  model.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

  # Train the top layer
  EPOCHS = 100
  model.fit(train_ds,
            epochs=EPOCHS,
            validation_data=val_ds,
            callbacks=callback)
  return model

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

def train_entire_model(base_model, model, train_ds, val_ds, EPOCHS, callback):
# Unfreeze the base model and train the entire model end-to-end with a low learning rate
  base_model.trainable = True
  model.summary()

  model.compile(
      optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
      loss=keras.losses.SparseCategoricalCrossentropy(),
      metrics=["accuracy"]
  )

  model.fit(train_ds,
            epochs=EPOCHS,
            validation_data=val_ds,
            callbacks=callback)

main()
