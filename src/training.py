from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import callbacks, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ReLU, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Lambda, ZeroPadding2D, concatenate, Average
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.backend as backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, CategoricalCrossentropy

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import cv2 as cv

import ranges_of_age as roa

def create_CNN(width, height, depth):
    input_shape = (height, width, depth)

    inputs = Input(shape=input_shape)

    filters = [64, 256, 1024]
    f_size = [5 ,5, 3]

    x = inputs

    for i, f in enumerate(filters):
        x = ZeroPadding2D(padding=2)(x)
        x = Conv2D(f, (f_size[i],f_size[i]), padding="valid")(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2,2))(x)

    x = Flatten()(x)
    x = Dense(100)(x)
    x = Dense(100)(x)
    x = Dropout(0.1)(x)
    x = Dense(1, activation='linear')(x)

    model = Model(inputs, x)

    return model

def get_images_list(dataframe,regresion=True):
    list = dataframe.to_numpy()

    image_list = []
    # img_names = ['_panorama_SDM.png', '_panorama_NDM.png', '_panorama_GNDM.png']
    img_names = ['_panorama_ext_X.png', '_panorama_ext_Y.png', '_panorama_ext_Z.png']
    # img_names = ['_panorama_SDM.png']

    if regresion:
        idx = 2
    else:
        idx = 1

    for t in list:
        imgs = []
        for i in np.arange(len(img_names)):
            img = t[0]+'/'+t[0]+img_names[i]
            imgs.append(img)
        imgs.append(t[idx])
        image_list.append(imgs)

    return np.array(image_list)

def rotate_image(image, img_shape, shuffle_pos):
    rot_image = np.zeros((img_shape[0],int(img_shape[1]/1.5),3))
    rot_image2 = np.zeros_like(image)
    img_max = int(img_shape[1]/1.5)
    rot_image[:,0:int(img_shape[1]/1.5),] = image[:,0:int(img_shape[1]/1.5):]
    rot_image2[:,:(img_max-shuffle_pos),] = rot_image[:,shuffle_pos:]
    rot_image2[:,(img_max-shuffle_pos):img_max,] = rot_image[:,:shuffle_pos]
    rot_image2[:,img_max:,] = rot_image2[:,0:int(img_max/2)]
    # cv.imshow('image',image)
    # cv.imshow('rot_image',rot_image2)
    # # image_opencv = cv.imread(img_path)
    # # cv.imshow('image_opencv',image_opencv)
    # cv.imwrite('./data_img/1_Dch/1_Dch_panorama_ext_Y_2.png', image)
    # cv.waitKey()
    # cv.destroyAllWindows()

    return rot_image2

def cached_img_read(img_path, img_shape, colormode, image_cache, shuffle_pos):
    if img_path not in image_cache.keys():
        # print(img_path)
        image = img_to_array(load_img(img_path, color_mode=colormode, target_size=img_shape, interpolation='bilinear')).astype(np.float32)
        image = image / 255.0
        if colormode == 'rgb':
            image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        image = rotate_image(image, img_shape, shuffle_pos)
        image_cache[img_path] = image
    
    return image_cache[img_path]

def read_images_gen(images, dir, img_shape, datagen, colormode, image_cache, augment):
    if colormode == "grayscale":
        X = np.zeros((len(images), img_shape[0], img_shape[1], 1))
    else:
        X = np.zeros((len(images), img_shape[0], img_shape[1], 3))

    for i, image_name in enumerate(images):
        image = cached_img_read(dir+image_name, img_shape, colormode, image_cache, augment[i])
        X[i] = datagen.standardize(image)
        
    return X
        

def image_generator(images, dir, batch_size, datagen, img_shape=(108,108), colormode="grayscale", shuffle=False):
    image_cache = {}

    while True:
        n_imgs = len(images)
        if shuffle:
            indexs = np.random.permutation(np.arange(n_imgs))
        else:
            indexs = np.arange(n_imgs)
        num_batches = n_imgs // batch_size
        
        for bid in range(num_batches):
            batch_idx = indexs[bid*batch_size:(bid+1)*batch_size]
            batch = [images[i] for i in batch_idx]
            augment = np.random.randint(0,int(img_shape[1]/2),batch_size)
            img1 = read_images_gen([b[0] for b in batch], dir, img_shape, datagen, colormode, image_cache, augment)
            img2 = read_images_gen([b[1] for b in batch], dir, img_shape, datagen, colormode, image_cache, augment)
            img3 = read_images_gen([b[2] for b in batch], dir, img_shape, datagen, colormode, image_cache, augment)
            label = np.array([b[3] for b in batch]).astype(np.float32)
            # label = np.array([b[1] for b in batch]).astype(np.float32)
            # print([img1, img2, img3], label)
            yield ([img1, img2, img3], label)
            # yield ([img1], label)

def mostrarEvolucion(hist, save=False, dyh=""):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    #   if save:
    #     name = "Loss_" + dyh
    #     plt.savefig(name)
    plt.show()


BATCH_SIZE = 1
EPOCHS = 100
IMG_SHAPE = (90,270)
REGRESSION = True

train_val = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

train, validation = train_test_split(train_val, test_size=0.25)

train = get_images_list(train,regresion=REGRESSION)
validation = get_images_list(validation,regresion=REGRESSION)
test = get_images_list(test,regresion=REGRESSION)

datagen = ImageDataGenerator()
print("[INFO]: Starting image load process...")
datagen_train = image_generator(train, "./data_img/", BATCH_SIZE, datagen, IMG_SHAPE, colormode='rgb')
print("[INFO]: Train data loaded...")
datagen_val = image_generator(validation, "./data_img/", BATCH_SIZE, datagen, IMG_SHAPE, colormode='rgb')
print("[INFO]: Validation data loaded...")
datagen_test = image_generator(test, "./data_img/", 1, datagen, IMG_SHAPE, colormode='rgb', shuffle=False)
print("[INFO]: Test data loaded...")

def mae(y_true, y_pred):
    #  return (y_true - y_pred) ** 2
    return abs(y_true - y_pred)

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

cnn_SDM = create_CNN(IMG_SHAPE[0],IMG_SHAPE[1],3)
cnn_NDM = create_CNN(IMG_SHAPE[0],IMG_SHAPE[1],3)
cnn_GNDM = create_CNN(IMG_SHAPE[0],IMG_SHAPE[1],3)

# combined_input = concatenate([cnn_SDM.output, cnn_NDM.output, cnn_GNDM.output])
x = Average()([cnn_SDM.output, cnn_NDM.output, cnn_GNDM.output])

# x = Average()([cnn_SDM.output, cnn_NDM.output, cnn_GNDM.output])

model = Model(inputs=[cnn_SDM.input, cnn_NDM.input, cnn_GNDM.input], outputs=x)

# model = cnn_SDM

# optimizer = SGD(momentum = 0.9)
optimizer = Adam()


callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)

label_weights = np.zeros(len(train))

model.compile(loss=MeanAbsoluteError(), optimizer=optimizer)

# model.compile(loss=CategoricalCrossentropy(), optimizer=optimizer, metrics=["accuracy"])

# model.compile(loss=MeanSquaredError(), optimizer=optimizer)


model.summary()

print('GPU name:',tf.test.gpu_device_name())



history = model.fit(
    datagen_train,
    # validation_data=datagen_val,
    epochs=EPOCHS,
    verbose=1,
    steps_per_epoch=(len(train) // BATCH_SIZE),
    sample_weight=label_weights,
    # validation_steps=(len(validation) // BATCH_SIZE),
    # callbacks=[callback]
)

model.evaluate(
    datagen_test,
    batch_size=1,
    verbose=1,
    steps=len(test)
)

prediction = model.predict(datagen_test,
    verbose=1,
    steps=len(test)
)

true = np.array([float(n) for n in test[:,3]])
pred = prediction.flatten()

stats_mae = abs(true-pred)
print('Max value:',np.max(stats_mae))
print('Min value:',np.min(stats_mae))
print('Median:',np.median(stats_mae))
print('Mean:',np.mean(stats_mae))
print('Std:',np.std(stats_mae))
print('----------------------------------')
stats_mse = (true-pred) ** 2
print('Max value:',np.max(stats_mse))
print('Min value:',np.min(stats_mse))
print('Median:',np.median(stats_mse))
print('Mean:',np.mean(stats_mse))
print('Std:',np.std(stats_mse))

def range_of_age(n,ranges):
    for i,r in enumerate(ranges):
        if n >= r[0] and n <= r[1]:
            return i


def precision_by_range(y_true, y_pred):

    prec_dic = {}
    for r in np.arange(len(roa.ranges)):
        prec_dic[r] = [[],[]]

    for yt, yp in zip(y_true, y_pred):
        r_age = range_of_age(yt, roa.ranges)
        # print(yt, yp, ranges[r_age])
        dif = abs(yt - yp)
        prec_dic[r_age][0].append(dif)
        prec_dic[r_age][1].append(yp)

    precision = {}
    for k in prec_dic.keys():
        values = np.array(prec_dic[k][0])
        mean = np.mean(values)
        std = np.std(values)
        value_mean = np.mean(np.array(prec_dic[k][1]))
        value_std = np.std(np.array(prec_dic[k][1]))
        precision[roa.ranges[k]] = [value_mean,value_std,mean,std]

    print(precision)


if(REGRESSION):
    precision_by_range(true, pred)
else:
    print(true,'\n',pred)


# mostrarEvolucion(history)