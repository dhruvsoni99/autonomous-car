import tensorflow as tf
import os
from datetime import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Lambda, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

tf.random.set_seed(123)

def preprocess_image(image, label):

    # Convert the image to float32 TensorFlow tensor
    image = tf.cast(image, tf.float32)
    
    # Convert the image to grayscale
    gray_image = tf.image.rgb_to_grayscale(image)
    
    # Replicate the grayscale image across 3 channels
    three_channel_gray = tf.repeat(gray_image, repeats=3, axis=-1)
    
    
    return three_channel_gray, label
    #return tf.cast(image, tf.float32), label

# Model definition with preprocessing included
def build_model(num_classes):

    # with tf.device('/cpu:0'):
    data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomBrightness(0.1, seed=123),
                                             tf.keras.layers.RandomContrast(0.1, seed=123),
                                            tf.keras.layers.RandomFlip('horizontal', seed=123),
                                            # tf.keras.layers.CenterCrop(224,224),
                                            # tf.keras.layers.RandomZoom(0.1, fill_mode='reflect', seed=123),
                                            # tf.keras.layers.RandomRotation(0.05, fill_mode='nearest', seed=123),

    ])

    base_model = tf.keras.applications.ResNet50(input_shape=(192, 192, 3), include_top=False, weights=None)
    base_model.trainable = True  # Freeze base model

    inputs = Input(shape=(192,192,3))
    aug_inputs = data_augmentation(inputs)
    x = preprocess_input(aug_inputs)  # Preprocessing for Resnet
    x = base_model(x, training=True)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.35)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.35)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)

    # optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer,
                  loss='categorical_focal_crossentropy',
                  metrics=['accuracy', F1Score(), tf.metrics.MeanSquaredError()])
    return model

# Fine-tuning function
def fine_tune_model(model):
    # Unfreeze all layers in base model
    print("Finetuning ...")

    model.layers[4].trainable = True
    for layer in model.layers[4].layers[:-15]:
        layer.trainable = False
    for layer in model.layers[4].layers[:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.000001)
    # Recompile the model with a lower learning rate
    model.compile(optimizer=optimizer,
                  loss='categorical_focal_crossentropy',
                  metrics=['accuracy', F1Score(), tf.metrics.MeanSquaredError()])
    model.summary()

finetuning = False

if __name__ == "__main__":
    directory = 'angle_class_data'
    train_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                labels='inferred',
                label_mode='categorical',
                #color_mode='rgb',
                color_mode='rgb',
                batch_size=128,
                image_size = (192,192),
                shuffle=True,
                seed=123,
                validation_split=0.2,
                subset="training").map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                labels='inferred',
                label_mode='categorical',
                color_mode='rgb',
                batch_size=32,
                image_size = (192,192),
                shuffle=True,
                seed=123,
                validation_split=0.2,
                subset="validation").map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)

    checkpoint_path = "models/untrained/angle/resnet_best"

    model_checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor='val_mean_squared_error',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    save_freq='epoch')

    if not finetuning:
        print("Base Learning")
        model = build_model(17)
        model.summary()

        history = model.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=[model_checkpoint_callback], verbose=1)

    # Load the best model for fine-tuning
    model = tf.keras.models.load_model(checkpoint_path)

    # Fine-tuning
    fine_tune_model(model)
    # Fine-tuning training with a smaller learning rate
    fine_tune_epochs = 10
    total_epochs = 10 + fine_tune_epochs  # Initial epochs + fine-tune epochs

    history_finetune = model.fit(
                            train_ds,
                            validation_data=val_ds,
                            epochs=total_epochs,
                            callbacks=[model_checkpoint_callback],
                            verbose=1)

    # # Evaluate the fine-tuned model
    # evaluate_model(model, valid_generator)s