import enum
from random import random
from re import S
from xml.etree.ElementInclude import include
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import callbacks, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ReLU, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Lambda, ZeroPadding2D, concatenate, Average
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, CategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import VGG16

from sklearn.model_selection import KFold

from keras.utils.vis_utils import plot_model

from collections import Counter

import scipy.ndimage as sci

import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import cv2 as cv

import random

import ranges_of_age as roa

import sys

from datetime import datetime

import functions as fn

COLORMODE = sys.argv[1]
VERBOSE = 2
RESNET = bool(int(sys.argv[2]))
MODEL = sys.argv[3]
mse = []
r2 = []
rmse = []
mae = []

if COLORMODE == 'rgb':
    IMG_SHAPE = (36,108)
    depth = 3
    if RESNET:
        cnn = fn.create_CNN_Resnet
        model_weights = './execution/'+str(MODEL)+'/model_rgb_resnet/model_rgb_resnet'
    else:
        cnn = fn.create_CNN
        model_weights = './execution/'+str(MODEL)+'/model_rgb_panoramacnn/model_rgb_panoramacnn'
elif COLORMODE == 'grayscale':
    IMG_SHAPE = (108,108)
    depth = 1
    cnn = fn.create_CNN
    model_weights = './execution/'+str(MODEL)+'/model_gray_panoramacnn/model_gray_panoramacnn'

cnn_0 = cnn(0,IMG_SHAPE[0],IMG_SHAPE[1],depth)
cnn_1 = cnn(1,IMG_SHAPE[0],IMG_SHAPE[1],depth)
cnn_2 = cnn(2,IMG_SHAPE[0],IMG_SHAPE[1],depth)

x = Average()([cnn_0.output, cnn_1.output, cnn_2.output])

model = Model(inputs=[cnn_0.input, cnn_1.input, cnn_2.input], outputs=x)

optimizer = Adam()

model.compile(loss=MeanAbsoluteError(), optimizer=optimizer, metrics=['mae'])
model.summary()

model.load_weights(model_weights)

test_df = pd.read_csv('./test.csv')
test = fn.get_images_list(test_df,COLORMODE,weights=False)
test_not_augment = fn.get_images_list(test_df,COLORMODE,augment=False,weights=False)
datagen = ImageDataGenerator()
datagen_test = fn.image_generator(test, "./data_img_norm/", 1, datagen, IMG_SHAPE, colormode=COLORMODE, shuffle=False, weights=False)
datagen_test_not_augment = fn.image_generator(test_not_augment, "./data_img_norm/", 1, datagen, IMG_SHAPE, colormode=COLORMODE, shuffle=False, weights=False)

prediction = model.predict(datagen_test, verbose=VERBOSE, steps=len(test))
true = np.array([float(n) for n in test[:,3]])
pred = prediction.flatten()

fn.show_stats(true,pred,metric_range='mse')

mse.append(mean_squared_error(true, pred))
r2.append(r2_score(true, pred)*100)
rmse.append(mean_squared_error(true, pred, squared= False))
mae.append(mean_absolute_error(true, pred))

kfold_stats_df = pd.DataFrame()
kfold_stats_df['MSE'] = mse
kfold_stats_df['RMSE'] = rmse
kfold_stats_df['MAE'] = mae
kfold_stats_df['R2'] = r2

print(kfold_stats_df.to_string())

prediction_not_augment = model.predict(datagen_test_not_augment, verbose=VERBOSE, steps=len(test_not_augment))
true = np.array([float(n) for n in test_not_augment[:,3]])
pred = prediction_not_augment.flatten()

fn.show_stats(true,pred,metric_range='mse')

mse.append(mean_squared_error(true, pred))
r2.append(r2_score(true, pred)*100)
rmse.append(mean_squared_error(true, pred, squared= False))
mae.append(mean_absolute_error(true, pred))

kfold_stats_df = pd.DataFrame()
kfold_stats_df['MSE'] = mse
kfold_stats_df['RMSE'] = rmse
kfold_stats_df['MAE'] = mae
kfold_stats_df['R2'] = r2

print(kfold_stats_df.to_string())