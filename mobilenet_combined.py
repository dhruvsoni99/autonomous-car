import os
import datetime
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import cv2
from utils import get_merged_df

import tensorflow as tf

import keras
from keras.models import Sequential, Model  
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop

print( f'tf.__version__: {tf.__version__}' )
print( f'keras.__version__: {keras.__version__}' )

from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

def my_imread(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def img_preprocess(image, label):

    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (192,192)) 

    # DATA AUGMENTATION -> IMAGE FLIP
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        label = 1 - label

    return image, label


def image_data_generator(image_paths, labels_dict, batch_size):
    while True:
        batch_images = []
        batch_angles = []
        batch_speeds = []
        
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image = my_imread(image_paths[random_index])

            angle_label = labels_dict['angle_output'][random_index]
            speed_label = labels_dict['speed_output'][random_index]
              
            image, angle_label = img_preprocess(image, angle_label)

            angle_label = int(round(angle_label * 16)) 
            batch_images.append(image)
            
            # do one hot category
            angle_one_hot = to_categorical(angle_label, num_classes=17) 
            batch_angles.append(angle_one_hot)
            
            # add speed label 
            batch_speeds.append(speed_label)
            
        batch_angles = np.array(batch_angles)
        batch_angles = batch_angles.reshape((batch_size, 17))
        yield (np.asarray(batch_images), {'angle_output': batch_angles, 'speed_output': np.array(batch_speeds)})

def mobile_net_classification_model():
    inputs = Input(shape=(192, 192, 3))
    x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(inputs)
    
    base_model = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_tensor=x)
    base_model.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)

    common = Dense(1024, activation='relu')(x)
    common = Dropout(0.5)(common)

    # angle prediction
    angle_branch = Dense(512, activation='relu')(common)
    angle_branch = tf.keras.layers.BatchNormalization()(angle_branch)
    angle_branch = Dropout(0.5)(angle_branch)
    angle_output = Dense(17, activation='softmax', name='angle_output')(angle_branch)

    # speed prediction
    speed_branch = Dense(512, activation='relu')(common)
    speed_branch = Dropout(0.5)(speed_branch)
    speed_output = Dense(1, activation='sigmoid', name='speed_output')(common) 


    model = Model(inputs=inputs, outputs=[angle_output, speed_output])
   
    # learning rate
    custom_lr = 0.001  
    optimizer = RMSprop(learning_rate=custom_lr)

    model.compile(optimizer='adam',
                  loss={'angle_output': 'categorical_crossentropy', 'speed_output': 'binary_crossentropy'},
                  metrics={'angle_output': 'accuracy', 'speed_output': 'accuracy'})

    return model

def predict(self, image):
        angles = np.arange(17)*5+50
  
        image = self.preprocess(image)
        
        pred_speed = self.combined_model.predict(image)[1][0][0]
        
        if pred_speed < 0.5:
            pred_speed = 0
        else:
            pred_speed = 1

        speed = pred_speed * 35

        pred_angle = self.combined_model.predict(image)[0][0]
        angle = np.argmax(pred_angle)
    
        predicted_angle = angles[angle]

        print('angle:', angle,'speed:', speed)
        return predicted_angle, speed
    

data_dir = 'training_data/training_data'
norm_csv_path = 'training_norm.csv'
cleaned_df = get_merged_df(data_dir, norm_csv_path)

angle_labels = cleaned_df['angle'].to_list()
speed_labels = cleaned_df['speed'].to_list()
image_paths = cleaned_df['image_path'].to_list()

X_train, X_valid, angle_train, angle_valid, speed_train, speed_valid = train_test_split(image_paths, angle_labels, speed_labels, test_size=0.3)

model = mobile_net_classification_model()

print(model.summary())

model_output_dir = '/home/psysm13/GOAT-3/autopilot/models/'  

# clean up log folder for tensorboard
log_dir_root = f'{model_output_dir}/logs'

tensorboard_callback = TensorBoard(log_dir_root, histogram_freq=1)

# path to save model
tfmodelpath = '/home/psysm13/GOAT-3/autopilot/models/404_Driver_Not_Found/combinedmodel_best/'

# early stopping in case the val loss does not improve
early_stopping = EarlyStopping(
monitor='val_loss',  
patience=10,         
verbose=1,           
restore_best_weights=True  
)

# create a checkpoint callback
model_checkpoint_callback = ModelCheckpoint(
    tfmodelpath,
    monitor='val_loss',     # monitor val loss
    verbose=10,              
    save_best_only=True,    
    save_weights_only=False, 
    mode='min',             
    save_freq='epoch',
    save_format='tf'      
    )

history = model.fit(
    image_data_generator(X_train, {'angle_output': angle_train, 'speed_output': speed_train}, batch_size=128),
    steps_per_epoch=len(X_train) // 128,
    epochs=10,
    validation_data = image_data_generator(X_valid, {'angle_output': angle_valid, 'speed_output': speed_valid}, batch_size=128),
    validation_steps=len(X_valid) // 128,
    verbose=1,
    callbacks=[early_stopping, model_checkpoint_callback, tensorboard_callback]
    )

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00001) 

# unfreeze top layers
for layer in model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer='adam',
    loss={'angle_output': 'categorical_crossentropy', 'speed_output': 'binary_crossentropy'},
    metrics={'angle_output': 'accuracy', 'speed_output': 'accuracy'})

# fine tune model 
history_fine = model.fit(
    image_data_generator(X_train, {'angle_output': angle_train, 'speed_output': speed_train}, batch_size=128),
    steps_per_epoch=len(X_train) // 128,
    epochs=10,  
    validation_data=image_data_generator(X_valid, {'angle_output': angle_valid, 'speed_output': speed_valid}, batch_size=128),
    validation_steps=len(X_valid) // 128,
    verbose=1,
    callbacks=[model_checkpoint_callback]  
    )

model.save('/home/psysm13/GOAT-3/autopilot/models/404_Driver_Not_Found/combinedmodel/', save_format='tf')
