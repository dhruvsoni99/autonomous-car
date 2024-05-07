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
from vit_keras import vit
import tensorflow_addons as tfa

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        tf.config.set_visible_devices(gpus[1], 'GPU')
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
    vit_model = vit.vit_b16(
        image_size = 160,
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        classes = num_classes)

    #base_model = tf.keras.applications.ResNet50(input_shape=(160, 160, 3), include_top=False, weights='imagenet')
    #base_model.trainable = False  # Freeze base model
    vit_model.trainable = False

    model = tf.keras.Sequential([
        vit_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation = tfa.activations.gelu),
        tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
        tf.keras.layers.Dense(num_classes, 'softmax')
    ],
    name = 'vision_transformer')
    '''
    inputs = Input(shape=(160,160,3))
    aug_inputs = data_augmentation(inputs)
    x = preprocess_input(aug_inputs)  # Preprocessing for Resnet
    x = vit_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.35)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.35)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    '''
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
    model.summary()

    # Check if the layer is a model (which has sub-layers)
    if hasattr(model.layers[4], 'layers'):
        # Unfreeze all layers except for the last 15
        for layer in model.layers[4].layers[:-15]:
            layer.trainable = True

        # Freeze BatchNormalization layers to maintain their statistics
        for layer in model.layers[4].layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    else:
        print("The layer does not contain sub-layers.")

    '''
    model.layers[4].trainable = True
    for layer in model.layers[4].layers[:-15]:
        layer.trainable = False
    for layer in model.layers[4].layers[:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    '''
            
    # optimizer = RMSprop(learning_rate=0.00001)  # Lower learning rate
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.000001)
    # Recompile the model with a lower learning rate
    model.compile(optimizer=optimizer,
                  loss='categorical_focal_crossentropy',
                  metrics=['accuracy', F1Score(), tf.metrics.MeanSquaredError(), tf.keras.metrics.TopKCategoricalAccuracy(k=3)])
    model.summary()

finetuning = True

if __name__ == "__main__":
    directory = 'angle_class_data'
    train_ds = tf.keras.utils.image_dataset_from_directory(
                directory,
                labels='inferred',
                label_mode='categorical',
                #color_mode='rgb',
                color_mode='rgb',
                batch_size=32,
                image_size = (160,160),
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
                image_size = (160,160),
                shuffle=True,
                seed=123,
                validation_split=0.2,
                subset="validation")
    
    class_names = val_ds.class_names  # Assuming train_ds is your training dataset
    print(class_names)
    val_ds = val_ds.map(preprocess_image).cache().prefetch(tf.data.AUTOTUNE)

    checkpoint_path = "models/angle/vit_best"

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

        history = model.fit(train_ds, epochs=1, validation_data=val_ds, callbacks=[model_checkpoint_callback], verbose=1)
    
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
                            epochs=1,
                            callbacks=[model_checkpoint_callback],
                            verbose=1)
    
    
    
    def evaluate_model(model):

        predictions = model.predict(val_ds)
        predicted_classes = np.argmax(predictions, axis=1)

        # Retrieve true labels
        # If your labels are one-hot encoded and test_ds was batched, unbatch and concatenate like this:
        true_labels = np.concatenate([y.numpy() for x, y in val_ds.unbatch()], axis=0)
        if true_labels.ndim > 1 and true_labels.shape[1] != 1:  # Check if labels are one-hot encoded
            true_classes = np.argmax(true_labels, axis=1)
        else:
            true_classes = true_labels.flatten()  # If labels are not one-hot encoded, simply flatten the array
            print(true_labels)
        
        # Check if true_labels needs argmax or is already integer-encoded
        if true_labels.ndim > 1:  # Assuming one-hot encoding if ndim > 1
            true_classes = np.argmax(true_labels, axis=1)
        else:  # Assuming labels are integer-encoded if ndim == 1
            true_classes = true_labels

        print(true_classes)
        print(predicted_classes)

        # Compute the confusion matrix
        cm = confusion_matrix(true_classes, predicted_classes)

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        plt.savefig('images/vitconfusionmatrix.png')

    # Evaluate the fine-tuned model
    model = tf.keras.models.load_model(checkpoint_path)
    evaluate_model(model)
    # evaluate_model(model, valid_generator)