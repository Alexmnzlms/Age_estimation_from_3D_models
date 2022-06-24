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

################################################################################
BATCH_SIZE = 32
EPOCHS = 100
# IMG_SHAPE = (90,270)
REGRESSION = True
COLORMODE = sys.argv[1]
VERBOSE = 2
# FOLDS = n_splits=df['Sample'].size
# FOLDS = 5
RESNET = bool(int(sys.argv[2]))
SEED = int(sys.argv[3])
OUTPUT = sys.argv[4]

if COLORMODE == 'rgb':
    IMG_SHAPE = (36,108)
    depth = 3
    if RESNET:
        cnn = fn.create_CNN_Resnet
    else:
        cnn = fn.create_CNN
elif COLORMODE == 'grayscale':
    IMG_SHAPE = (108,108)
    depth = 1
    cnn = fn.create_CNN

# df = pd.read_csv('./dataset.csv')
# calculate_weights(df)
# df_list = get_images_list(df,COLORMODE)
# if COLORMODE == 'rgb':
#     load_image_normalize_rgb(df_list, "./data_img/", IMG_SHAPE)
# elif COLORMODE == 'grayscale':
#     load_image_normalize_grayscale(df_list, "./data_img/", IMG_SHAPE)

# kf = KFold(FOLDS,shuffle=True)
# msew = []
mse = []
# r2w = []
r2 = []
# rmsew = []
rmse = []
# maew = []
mae = []
fold = 1

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

################################################################################
cnn_0 = cnn(0,IMG_SHAPE[0],IMG_SHAPE[1],depth)
cnn_1 = cnn(1,IMG_SHAPE[0],IMG_SHAPE[1],depth)
cnn_2 = cnn(2,IMG_SHAPE[0],IMG_SHAPE[1],depth)

x = Average()([cnn_0.output, cnn_1.output, cnn_2.output])

model = Model(inputs=[cnn_0.input, cnn_1.input, cnn_2.input], outputs=x)

# optimizer = RMSprop()
optimizer = Adam()
# optimizer = SGD(momentum = 0.9)

checkpoint_filepath = './checkpoint/'

# model.compile(loss=MeanSquaredError(), optimizer=optimizer, metrics=['mae'])
model.compile(loss=MeanAbsoluteError(), optimizer=optimizer, metrics=['mae'])
model.summary()

weights = model.get_weights()

if RESNET:
    print('FINE TUNNING')
    cnn_0 = fn.create_CNN_Resnet(0,IMG_SHAPE[0],IMG_SHAPE[1],depth,True)
    cnn_1 = fn.create_CNN_Resnet(1,IMG_SHAPE[0],IMG_SHAPE[1],depth,True)
    cnn_2 = fn.create_CNN_Resnet(2,IMG_SHAPE[0],IMG_SHAPE[1],depth,True)
    x = Average()([cnn_0.output, cnn_1.output, cnn_2.output])

    model_ft = Model(inputs=[cnn_0.input, cnn_1.input, cnn_2.input], outputs=x)

    optimizer_ft = Adam(1e-5)

    model_ft.compile(loss=MeanAbsoluteError(), optimizer=optimizer_ft, metrics=['mae'])
    model_ft.summary()

################################################################################

# for train_index, test_index in kf.split(df):
#     tf.keras.backend.clear_session()

print('Data:', COLORMODE)
print('Batch:', BATCH_SIZE)
print('Epochs:', EPOCHS)
print('Seed:', SEED)
if RESNET:
    print('RESNET50')
else:
    print('PANORAMA-CNN')
# print("\nTRAIN:", train_index, "\nTEST:", test_index)
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print('[INFO]: Training -', dt_string)
# train = df.iloc[train_index].copy()
# test = df.iloc[test_index].copy()
train_df = pd.read_csv('./train.csv')
validation_df = pd.read_csv('./validation.csv')
test_df = pd.read_csv('./test.csv')
# calculate_weights(train)
# calculate_weights(test)
# print(train)
# print(test)
train = fn.get_images_list(train_df,COLORMODE)
validation = fn.get_images_list(validation_df,COLORMODE,weights=False)
test = fn.get_images_list(test_df,COLORMODE,weights=False)
test_not_augment = fn.get_images_list(test_df,COLORMODE,augment=False,weights=False)
datagen = ImageDataGenerator()
print("[INFO]: Starting image load process...")
datagen_train = fn.image_generator(train, "./data_img_norm/", BATCH_SIZE, datagen, IMG_SHAPE, colormode=COLORMODE)
print("[INFO]: Train data loaded...")
datagen_val = fn.image_generator(validation, "./data_img_norm/", BATCH_SIZE, datagen, IMG_SHAPE, colormode=COLORMODE, weights=False)
print("[INFO]: Validation data loaded...")
datagen_test = fn.image_generator(test, "./data_img_norm/", 1, datagen, IMG_SHAPE, colormode=COLORMODE, shuffle=False, weights=False)
datagen_test_not_augment = fn.image_generator(test_not_augment, "./data_img_norm/", 1, datagen, IMG_SHAPE, colormode=COLORMODE, shuffle=False, weights=False)
print("[INFO]: Test data loaded...")

# model.set_weights(weights)

early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(EPOCHS*0.2))
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=VERBOSE)

history = model.fit(
    datagen_train,
    validation_data=datagen_val,
    epochs=EPOCHS,
    verbose=VERBOSE,
    steps_per_epoch=(len(train) // BATCH_SIZE),
    validation_steps=(len(validation) // BATCH_SIZE),
    callbacks=[early_stop_callback, model_checkpoint_callback]
)

model.load_weights(checkpoint_filepath)

# score = model.evaluate(datagen_test, verbose=VERBOSE, steps=len(test))
# print(f'Test loss: {score[0]} / Test mae: {score[1]}')

prediction = model.predict(datagen_test, verbose=VERBOSE, steps=len(test))
true = np.array([float(n) for n in test[:,3]])
# weight = np.array([float(n) for n in test[:,4]])
pred = prediction.flatten()

fn.show_stats(true,pred)

# print('MSE:', mean_squared_error(true,pred, squared=False, sample_weight=weight))
# print('R2:', r2_score(true,pred, sample_weight=weight))
# print('EVS:', explained_variance_score(true,pred, sample_weight=weight))

# msew.append(mean_squared_error(true, pred, sample_weight=weight))
mse.append(mean_squared_error(true, pred))
# r2w.append(r2_score(true, pred, sample_weight=weight)*100)
r2.append(r2_score(true, pred)*100)
# rmsew.append(mean_squared_error(true, pred, squared= False ,sample_weight=weight))
rmse.append(mean_squared_error(true, pred, squared= False))
# maew.append(mean_absolute_error(true, pred,sample_weight=weight))
mae.append(mean_absolute_error(true, pred))

prediction_not_augment = model.predict(datagen_test_not_augment, verbose=VERBOSE, steps=len(test_not_augment))
true = np.array([float(n) for n in test_not_augment[:,3]])
pred = prediction_not_augment.flatten()

fn.show_stats(true,pred)

mse.append(mean_squared_error(true, pred))
r2.append(r2_score(true, pred)*100)
rmse.append(mean_squared_error(true, pred, squared= False))
mae.append(mean_absolute_error(true, pred))


kfold_stats_df = pd.DataFrame()
kfold_stats_df['MSE'] = mse
kfold_stats_df['RMSE'] = rmse
kfold_stats_df['MAE'] = mae
kfold_stats_df['R2'] = r2
# kfold_stats_df['MSEW'] = msew
# kfold_stats_df['RMSEW'] = rmsew
# kfold_stats_df['MAEW'] = maew
# kfold_stats_df['R2W'] = r2w

print(kfold_stats_df.to_string())

if RESNET:
    print('FINE TUNNING')
    model_ft.load_weights(checkpoint_filepath)

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=int(EPOCHS*0.2))
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=VERBOSE)

    history = model_ft.fit(
        datagen_train,
        validation_data=datagen_val,
        epochs=EPOCHS,
        verbose=VERBOSE,
        steps_per_epoch=(len(train) // BATCH_SIZE),
        validation_steps=(len(validation) // BATCH_SIZE),
        callbacks=[early_stop_callback, model_checkpoint_callback]
    )

    model_ft.load_weights(checkpoint_filepath)

    # datagen_test = fn.image_generator(test, "./data_img_norm/", 1, datagen, IMG_SHAPE, colormode=COLORMODE, shuffle=False, augment=False, weights=False)

    # score = model.evaluate(datagen_test, verbose=VERBOSE, steps=len(test))
    # print(f'Test loss: {score[0]} / Test mae: {score[1]}')

    prediction = model_ft.predict(datagen_test, verbose=VERBOSE, steps=len(test))
    true = np.array([float(n) for n in test[:,3]])
    pred = prediction.flatten()

    fn.show_stats(true,pred)

    mse.append(mean_squared_error(true, pred))
    r2.append(r2_score(true, pred)*100)
    rmse.append(mean_squared_error(true, pred, squared= False))
    mae.append(mean_absolute_error(true, pred))

    prediction_not_augment = model_ft.predict(datagen_test_not_augment, verbose=VERBOSE, steps=len(test_not_augment))
    true = np.array([float(n) for n in test_not_augment[:,3]])
    pred = prediction_not_augment.flatten()

    fn.show_stats(true,pred)

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
    model_ft.save_weights(OUTPUT+'/model_rgb_resnet/model_rgb_resnet')

else:
    if COLORMODE == 'rgb':
        folder = OUTPUT + '/model_rgb_panoramacnn/model_rgb_panoramacnn'
    elif COLORMODE == 'grayscale':
        folder = OUTPUT + '/model_gray_panoramacnn/model_gray_panoramacnn'
    model.save_weights(folder)

print('################################################################################')
# msew.append(np.mean(msew))
mse.append(np.mean(mse))
# r2w.append(np.mean(r2w))
r2.append(np.mean(r2))
# rmsew.append(np.mean(rmsew))
rmse.append(np.mean(rmse))
# maew.append(np.mean(maew))
mae.append(np.mean(mae))

kfold_stats_df = pd.DataFrame()
kfold_stats_df['MSE'] = mse
kfold_stats_df['RMSE'] = rmse
kfold_stats_df['MAE'] = mae
kfold_stats_df['R2'] = r2
# kfold_stats_df['MSEW'] = msew
# kfold_stats_df['RMSEW'] = rmsew
# kfold_stats_df['MAEW'] = maew
# kfold_stats_df['R2W'] = r2w

print('\n')
print('Data:', COLORMODE)
print('Batch:', BATCH_SIZE)
print('Epochs:', EPOCHS)
print('Seed:', SEED)
if RESNET:
    print('RESNET50')
else:
    print('PANORAMA-CNN')

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print(dt_string)

print(kfold_stats_df.to_string())

# train = pd.read_csv('./train.csv')
# test = pd.read_csv('./test.csv')

# # train, validation = train_test_split(train_val, test_size=0.0)

# train = get_images_list(train,regresion=REGRESSION)
# # validation = get_images_list(validation,regresion=REGRESSION)
# test = get_images_list(test,regresion=REGRESSION)

# datagen = ImageDataGenerator()
# print("[INFO]: Starting image load process...")
# datagen_train = image_generator(train, "./data_img/", BATCH_SIZE, datagen, IMG_SHAPE, colormode=COLORMODE)
# print("[INFO]: Train data loaded...")
# # datagen_val = image_generator(validation, "./data_img/", 1, datagen, IMG_SHAPE, colormode='rgb')
# # print("[INFO]: Validation data loaded...")
# datagen_test = image_generator(test, "./data_img/", 1, datagen, IMG_SHAPE, colormode=COLORMODE, shuffle=False)
# print("[INFO]: Test data loaded...")

# # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# if COLORMODE == 'rgb':
#     cnn_SDM = create_CNN(0,IMG_SHAPE[0],IMG_SHAPE[1],3)
#     cnn_NDM = create_CNN(1,IMG_SHAPE[0],IMG_SHAPE[1],3)
#     cnn_GNDM = create_CNN(2,IMG_SHAPE[0],IMG_SHAPE[1],3)
# elif COLORMODE == 'grayscale':
#     cnn_SDM = create_CNN(0,IMG_SHAPE[0],IMG_SHAPE[1],1)
#     cnn_NDM = create_CNN(1,IMG_SHAPE[0],IMG_SHAPE[1],1)
#     cnn_GNDM = create_CNN(2,IMG_SHAPE[0],IMG_SHAPE[1],1)

# # combined_input = concatenate([cnn_SDM.output, cnn_NDM.output, cnn_GNDM.output])
# # x = Dense(1,activation='ReLU')(combined_input)

# x = Average()([cnn_SDM.output, cnn_NDM.output, cnn_GNDM.output])

# print('-----------------------------------------------------------------------')
# print(cnn_SDM.input)
# print(cnn_NDM.input)
# print(cnn_GNDM.input)

# model = Model(inputs=[cnn_SDM.input, cnn_NDM.input, cnn_GNDM.input], outputs=x)


# # optimizer = SGD(momentum = 0.9)
# optimizer = Adam()

# callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

# # model.compile(loss=MeanAbsoluteError(), optimizer=optimizer)

# # model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=["accuracy"])

# model.compile(loss=MeanSquaredError(), optimizer=optimizer, metrics=['mae'])

# model.summary()

# print('GPU name:',tf.test.gpu_device_name())



# history = model.fit(
#     datagen_train,
#     # validation_data=datagen_val,
#     epochs=EPOCHS,
#     verbose=1,
#     steps_per_epoch=(len(train) // BATCH_SIZE),
#     # validation_steps=(len(validation) // 1),
#     callbacks=[callback]
# )

# # model.evaluate(
# #     datagen_test,
# #     batch_size=1,
# #     verbose=1,
# #     steps=len(test)
# # )


# prediction = model.predict(datagen_test,
#     verbose=1,
#     steps=len(test)
# )

# true = np.array([float(n) for n in test[:,3]])
# pred = prediction.flatten()

# show_stats(true,pred,"result_training_test")

# # for i,l in enumerate(model.layers):
# #     for w in l.weights:
# #         print(i,w.shape)

# def show_filter(filters):
#     for f in filters:
#         # print(model.layers[f].weights[0])
#         filter = model.layers[f].weights[0]
#         for i,img in enumerate(np.split(filter,64,axis=3)):
#             img = np.squeeze(img, axis=3)
#             # img = np.swapaxes(img, 0,2)
#             print(img, img.shape)
#             cv.imwrite('./filters/6_'+str(i)+'.png',(img - img.min()) / (img.max() - img.min())*255.0)
#         # for d in filter:
#         #     print(d.shape)

#         images = [np.squeeze(img, axis=3) for img in np.split(filter,64,axis=3)]

#         import matplotlib.pyplot as plt
#         fig, axes = plt.subplots(8,8, figsize=(5,5))
#         for i,ax in enumerate(axes.flat):
#             img = images[i]
#             ax.imshow((img - img.min()) / (img.max() - img.min()))
#             ax.axis('off')
#     plt.show()

# # show_filter([6,7,8])

# # mostrarEvolucion(history)