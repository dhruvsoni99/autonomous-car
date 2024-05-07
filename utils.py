import os
import cv2
import pandas as pd
import shutil
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D, Lambda, GaussianNoise, BatchNormalization, Flatten, GlobalMaxPooling2D, Reshape, Layer, MaxPooling2D

def get_merged_df(data_dir, norm_csv_path):
     
     # get normalised data
     df = pd.read_csv(norm_csv_path)

     # init lists to store data
     image_id = []
     image_path = []
     image_array = []
     file_size = []

     # list files in the directory
     file_list = os.listdir(data_dir)

     # for each file in directory
     for filename in file_list:
         
         # read image 
         im = cv2.imread(os.path.join(data_dir, filename))

         # append data to lists
         image_id.append(int(filename.split('.')[0]))
         image_array.append(im)
         image_path.append(os.path.join(data_dir, filename))
         file_size.append(os.path.getsize(os.path.join(data_dir, filename)))

     # create df for data
     data = {
         'image_id': image_id,
         'image': image_array,
         'image_path': image_path,
         'file_size': file_size
     }
     df_image = pd.DataFrame(data)

     # merge df with normalised data
     merged_df = pd.merge(df, df_image, how='left', on='image_id')

     # remove speeds greater than one
     cleaned_df = merged_df[merged_df['speed'] <= 1]

     #  return merged df
     return cleaned_df
