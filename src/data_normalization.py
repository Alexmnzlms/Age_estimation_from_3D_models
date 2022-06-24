import pandas as pd
import numpy as np
from zmq import EVENT_CLOSE_FAILED
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2 as cv
from collections import Counter
import scipy.ndimage as sci

import functions as fn
# IMG_SHAPE_RGB = (36,108)
# IMG_SHAPE_GRAY = (108,108)
IMG_SHAPE_RGB = (36,108)
IMG_SHAPE_GRAY = (108,108)

print('Estandarizando train...')
train = pd.read_csv('./train.csv')
train_list_rgb = fn.get_images_list(train,'rgb')
train_list_gray = fn.get_images_list(train,'grayscale')

norm_params_rgb = fn.load_image_normalize_rgb(train_list_rgb, "./data_img/", IMG_SHAPE_RGB)
norm_params_gray = fn.load_image_normalize_grayscale(train_list_gray, "./data_img/", IMG_SHAPE_GRAY)

print('Estandarizando validacion...')
validation = pd.read_csv('./validation.csv')
validation_list_rgb = fn.get_images_list(validation,'rgb', augment=False, weights=False)
validation_list_gray = fn.get_images_list(validation,'grayscale', augment=False, weights=False)

fn.load_image_normalize_rgb(validation_list_rgb, "./data_img/", IMG_SHAPE_RGB, fit_data=False, list_values=norm_params_rgb)
fn.load_image_normalize_grayscale(validation_list_gray, "./data_img/", IMG_SHAPE_GRAY, fit_data=False, list_values=norm_params_rgb)

print('Estandarizando test...')
test = pd.read_csv('./test.csv')
test_list_rgb = fn.get_images_list(test,'rgb', augment=False, weights=False)
test_list_gray = fn.get_images_list(test,'grayscale', augment=False, weights=False)

fn.load_image_normalize_rgb(test_list_rgb, "./data_img/", IMG_SHAPE_RGB, fit_data=False, list_values=norm_params_rgb)
fn.load_image_normalize_grayscale(test_list_gray, "./data_img/", IMG_SHAPE_GRAY, fit_data=False, list_values=norm_params_rgb)