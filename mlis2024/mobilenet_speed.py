import os
from datetime import datetime
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# sklearn
from sklearn.model_selection import train_test_split
from utils import get_merged_df

# tensorflow
import tensorflow as tf
import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from keras.models import Sequential, Model  # V2 is tensorflow.keras.xxxx, V1 is keras.xxx
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import F1Score, AUC, CategoricalAccuracy, BinaryAccuracy
from tensorflow.keras.optimizers import RMSprop, Adam

print( f"tf.__version__: {tf.__version__}" )
# print( f"keras.__version__: {keras.__version__}" )

import cv2
from PIL import Image

timestamp = datetime.now().strftime('%Y%m%d')

data_dir = 'training_data/training_data'
norm_csv_path = 'training_norm.csv'
cleaned_df = get_merged_df(data_dir, norm_csv_path)

X_train, X_valid, y_train, y_valid = train_test_split(cleaned_df['image_path'].to_list(), cleaned_df['speed'].to_list(), test_size=0.3)

# X_train, X_valid, angle_train, angle_valid, speed_train, speed_valid = train_test_split(image_paths, angle_labels, speed_labels, test_size=0.3)
print("Training data: %d\nValidation data: %d" % (len(X_train), len(X_valid)))

def my_imread(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def img_preprocess(image):
    # height, _, _ = image.shape
    # image = image[int(height/2):,:,:]  # remove top half of the image, as it is not relavant for lane following
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  # Nvidia model said it is best to use YUV color space
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (224,224)) # input image size (200,66) Nvidia model
    # image = image / 255 # normalizing, the processed image becomes black for some reason.  do we need this?
    # image = (image - 127.5) / 127.5
    return image

def image_data_generator(image_paths, speed_labels, batch_size):
    while True:
        batch_images = []
        batch_speeds = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image = my_imread(image_paths[random_index])
            speed_label = speed_labels[random_index]

            image = img_preprocess(image)
            batch_images.append(image)

            batch_speeds.append(speed_label)

        yield( np.asarray(batch_images), np.asarray(batch_speeds))

def mobile_net_classification_model():
    base_model = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3))
    base_model.trainable = False
    
    inputs = Input(shape=(224, 224, 3))
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(inputs)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Common part of the model
    common = Dense(1024, activation='relu')(x)
    common = Dropout(0.3)(common)

    # Branch for the angle prediction (multi-class classification)
    speed_output = Dense(1, activation='sigmoid', name='speed_output')(common) # Binary classification for speed

    model = Model(inputs=inputs, outputs=speed_output)

    # Create an RMSprop optimizer with a custom learning rate
    custom_lr = 0.001  # Example custom learning rate
    # optimizer = RMSprop(learning_rate=custom_lr)
    optimizer = Adam(learning_rate=custom_lr)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics='accuracy')

    return model

# model = nvidia_model()
model = mobile_net_classification_model()
model.summary()

model_output_dir = 'models/speed'

# start Tensorboard before model fit, so we can see the epoch tick in Tensorboard
# Jupyter Notebook embedded Tensorboard is a new feature in TF 2.0!!  

# clean up log folder for tensorboard
log_dir_root = f'{model_output_dir}/logs'
#!rm -rf $log_dir_root

tensorboard_callback = TensorBoard(log_dir_root, histogram_freq=1)

# Specify the file path where you want to save the model
filepath = f'models/speed/{timestamp}'

# Create the ModelCheckpoint callback
model_checkpoint_callback = ModelCheckpoint(
    filepath,
    monitor='val_loss',     # Monitor validation loss
    verbose=1,              # Log a message each time the callback saves the model
    save_best_only=True,    # Only save the model if 'val_loss' has improved
    save_weights_only=False, # Only save the weights of the model
    mode='min',             # 'min' means the monitored quantity should decrease
    save_freq='epoch'       # Check every epoch
)

history = model.fit(
    image_data_generator(X_train, y_train, batch_size=128),
    steps_per_epoch=len(X_train) // 128,
    epochs=10,
    validation_data = image_data_generator(X_valid, y_valid, batch_size=128),
    validation_steps=len(X_valid) // 128,
    verbose=1,
    shuffle=1,
    callbacks=[model_checkpoint_callback, tensorboard_callback]
)

# Load the model
model = load_model(filepath)

# optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
optimizer = Adam(learning_rate=0.00001)

# Unfreeze the top 20 layers of the model
for layer in model.layers[2].layers[-20:]:
    layer.trainable = True

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Step 3: Continue training the model
history_fine = model.fit(
    image_data_generator(X_train, y_train, batch_size=128),
    steps_per_epoch=len(X_train) // 128,
    epochs=10,  # You can adjust the number of epochs for fine-tuning
    validation_data=image_data_generator(X_valid, y_valid, batch_size=128),
    validation_steps=len(X_valid) // 128,
    verbose=1,
    callbacks=[model_checkpoint_callback]  # Assuming this callback is already defined
)
