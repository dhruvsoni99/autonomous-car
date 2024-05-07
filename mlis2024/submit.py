from datetime import datetime
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def preprocess(image_path):
    image = tf.keras.utils.load_img(image_path)
    im = tf.cast(image, tf.float32)
    im = tf.image.resize(im, [192, 192])
    im = tf.expand_dims(im, axis=0)
    return im

# def preprocess(image_path):
#     img = image.load_img(image_path, target_size=(192, 192))
#     img_array = image.img_to_array(img)
#     img_array_expanded_dims = np.expand_dims(img_array, axis=0)
#     return img_array_expanded_dims

def predict_speed(image_path, model):
    image = preprocess(image_path)
    speed = model.predict(image)[0]
    # Assuming binary classification for speed, adjust as necessary
    speed_pred = np.round(speed[0]).astype(int)
    return speed_pred

def predict_angle(image_path, model):
    angles = [100.0,105.0,110.0,115.0,120.0,125.0,130.0,50.0,55.0,60.0,65.0,70.0,75.0,80.0,85.0,90.0,95.0]
    image = preprocess(image_path)

    angle = model.predict(image)[0]
    pred_angle = angles[np.argmax(angle)]
    
    return (pred_angle-50)/80

timestamp = datetime.now().strftime('%Y%m%d')

angle_path = 'models/angle/resnet_best'
speed_path = 'models/speed/resnet_speed_best_20240322'


angle_model = load_model(angle_path)
speed_model = load_model(speed_path)

# TO CREATE PREDICTIONS OF THE TEST DATA
test_dir = os.path.abspath('test_data/test_data')
file_list = os.listdir(test_dir)
predictions = []

for file_name in sorted(file_list, key=lambda x: int(x.split('.')[0])):
    img_path = os.path.join(test_dir, file_name)
    image_id = int(file_name.split('.')[0])
    
    angle_pred = predict_angle(img_path, angle_model)
    speed_pred = predict_speed(img_path, speed_model)
    
    print(f"Image ID: {image_id}, Angle: {angle_pred}, Speed: {speed_pred}")
    predictions.append([image_id, angle_pred, speed_pred])

df_pred = pd.DataFrame(predictions, columns=['image_id', 'angle', 'speed'])
df_pred.to_csv(f'submission_{timestamp}.csv', index=False)