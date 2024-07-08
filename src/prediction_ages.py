from random import random
from re import S
from xml.etree.ElementInclude import include
import tensorflow as tf
from tensorflow.keras import callbacks, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ReLU, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Lambda, ZeroPadding2D, concatenate, Average
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError, CategoricalCrossentropy
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import VGG16
from keras.utils.vis_utils import plot_model
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import sys
import functions as fn
import argparse
import json
import logging
import os

def main(test_path, data_path, color_mode, img_shape, depth, cnn, verbose_level, model_folder, model_name, output):
    trained_models = os.listdir(model_folder)
    trained_models = [item for item in trained_models if os.path.isdir(os.path.join(model_folder,item))]

    if model_name == "model_rgb_resnet" and "232011" in trained_models:
        trained_models.remove("232011")

    print(trained_models)


    for model_code in trained_models:
        pretrained_model = os.path.join(model_folder, model_code, model_name, model_name)

        cnn_0 = cnn(0,img_shape[0],img_shape[1],depth)
        cnn_1 = cnn(1,img_shape[0],img_shape[1],depth)
        cnn_2 = cnn(2,img_shape[0],img_shape[1],depth)

        ###############################################################################

        x = [cnn_0.output, cnn_1.output, cnn_2.output]

        model = Model(inputs=[cnn_0.input, cnn_1.input, cnn_2.input], outputs=x)

        optimizer = Adam()

        model.compile(loss=MeanAbsoluteError(), optimizer=optimizer, metrics=['mae'])
        model.summary()

        print(pretrained_model)
        model.load_weights(pretrained_model)

        test_df = pd.read_csv(test_path)
        test = fn.get_images_list(test_df,color_mode,weights=False)
        test_not_augment = fn.get_images_list(test_df,color_mode,augment=False,weights=False)
        datagen = ImageDataGenerator()
        # datagen_test = fn.image_generator(test, data_path, 1, datagen, img_shape, colormode=color_mode, shuffle=False, weights=False)
        datagen_test_not_augment = fn.image_generator(test_not_augment, data_path, 1, datagen, img_shape, colormode=color_mode, shuffle=False, weights=False)

        prediction_not_augment_split = model.predict(datagen_test_not_augment, verbose=verbose_level, steps=len(test_not_augment))
        true = np.array([float(n) for n in test_not_augment[:,3]])

        ###############################################################################

        x = Average()([cnn_0.output, cnn_1.output, cnn_2.output])

        model = Model(inputs=[cnn_0.input, cnn_1.input, cnn_2.input], outputs=x)

        optimizer = Adam()

        model.compile(loss=MeanAbsoluteError(), optimizer=optimizer, metrics=['mae'])

        model.load_weights(pretrained_model)

        test_df = pd.read_csv(test_path)
        test = fn.get_images_list(test_df,color_mode,weights=False)
        test_not_augment = fn.get_images_list(test_df,color_mode,augment=False,weights=False)
        datagen = ImageDataGenerator()
        # datagen_test = fn.image_generator(test, data_path, 1, datagen, img_shape, colormode=color_mode, shuffle=False, weights=False)
        datagen_test_not_augment = fn.image_generator(test_not_augment, data_path, 1, datagen, img_shape, colormode=color_mode, shuffle=False, weights=False)

        prediction_not_augment = model.predict(datagen_test_not_augment, verbose=verbose_level, steps=len(test_not_augment))
        true = np.array([float(n) for n in test_not_augment[:,3]])
        pred = prediction_not_augment.flatten()
        names = np.array([n.split("/")[0].split("_") for n in test_not_augment[:,0]]).T

        ###############################################################################

        prediction_dict = {
            "SAMPLE": names[0],
            "SIDE": names[1],
            "TRUE": true,
            "X AXIS": prediction_not_augment_split[0].flatten(),
            "Y AXIS": prediction_not_augment_split[1].flatten(),
            "Z AXIS": prediction_not_augment_split[2].flatten(),
            "AXIS AVERAGE": np.mean([ prediction.flatten() for prediction in prediction_not_augment_split ], axis = 0),
            "COMBINED OUTPUT": pred,
            "ERROR": true-pred
        }

        dataframe = pd.DataFrame.from_dict(prediction_dict)

        print(os.path.join(output, "age_prediction_"+model_code+"_"+model_name+".csv"))
        dataframe.to_csv(os.path.join(output, "age_prediction_"+model_code+"_"+model_name+".csv"), index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-p",
        "--parameters",
        help="JSON file with parameters configuration",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    json_file = open(args.parameters, "r")
    args = json.load(json_file)
    json_file.close()

    if args["color_mode"] == 'rgb':
        img_shape = (36,108)
        depth = 3
        if args["resnet"]:
            cnn = fn.create_CNN_Resnet
        else:
            cnn = fn.create_CNN
    elif args["color_mode"] == 'grayscale':
        img_shape = (108,108)
        depth = 1
        cnn = fn.create_CNN
    else:
        raise RuntimeError("Color mode option unknown")

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    tf.random.set_seed(args["seed"])

    log_level = logging.WARNING
    if args["verbose"] == 0:
        log_level = logging.WARNING
    elif args["verbose"] == 1:
        log_level = logging.INFO
    elif args["verbose"] == 2:
        log_level = logging.DEBUG
    else:
        logging.warning('Log level not recognised. Using WARNING as default')

    logging.getLogger().setLevel(log_level)

    logging.warning("Verbose level set to {}".format(logging.root.level))

    main(
        args["test"],
        args["data"],
        args["color_mode"], 
        img_shape, 
        depth, 
        cnn,
        args["verbose_keras"],
        args["model_folder"],
        args["model_name"],
        args["output"]
        )