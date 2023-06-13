import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras.models import Sequential
from keras import layers

TRAIN_PATH = "D:/DataScience/Lab/Image Classification/Classification_data/train/full/"
TEST_PATH = "D:/DataScience/Lab/Image Classification/Classification_data/test/full/"

batch_size = 32
img_height = 150
img_width = 150

train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = keras.utils.image_dataset_from_directory(
    TEST_PATH,
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),]
)

base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet
    input_shape=(150, 150, 3),
    include_top=False,
    classes=6
)

# Freeze the base_model
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
"""
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(None, 150, 150, 3)]     0

 sequential (Sequential)     (None, 150, 150, 3)       0

 rescaling (Rescaling)       (None, 150, 150, 3)       0

 xception (Functional)       (None, 5, 5, 2048)        20861480

 global_average_pooling2d (G  (None, 2048)             0
 lobalAveragePooling2D)

 dropout (Dropout)           (None, 2048)              0

 dense (Dense)               (None, 1)                 2049

=================================================================
Total params: 20,863,529
Trainable params: 2,049
Non-trainable params: 20,861,480
"""
# Initialize early stopping to prevent overfitting
callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=1,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True,
    start_from_epoch=0
)

model.compile(optimizer="adam",
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

# Train the top layer
EPOCHS = 10
model.fit(train_ds,
          epochs=EPOCHS,
          validation_data=val_ds,
          callbacks=callback)

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