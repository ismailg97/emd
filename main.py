import builtins
import math

import numpy as np
import pandas as pd
import sklearn.metrics
import tensorflow as tf
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, recall_score
import keras
import keras.utils as ut
from typing import ClassVar, Callable, Union, List
from loss_functions.emd import earth_mover_distance, self_guided_earth_mover_distance, \
    approximate_earth_mover_distance, GroundDistanceManager, EmdWeightHeadStart
from data_handlers import pre_processing
from models import operations

from scipy.io import loadmat
from src.utils import get_meta, get_age

import ktrain
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image as pil

from models.model_from_paper import PaperModel
from models.standard_model import MyModel


def _compile_model(
        model: keras.Model,
        loss_function: Callable,
        ground_distance_path: Path,
        **loss_function_kwargs
):
    if loss_function == self_guided_earth_mover_distance:
        model.emd_weight_head_start = EmdWeightHeadStart()
        model.ground_distance_manager = GroundDistanceManager(ground_distance_path)
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #    learning_rate=0.001,
    #    decay_steps=429,
    #    decay_rate=0.995
    #)
    model.compile(
        loss=loss_function(
            model=model,
            **loss_function_kwargs
        ),
        optimizer=tf.keras.optimizers.Adam(
            #learning_rate=lr_schedule,
            learning_rate=0.001,
            # nesterov=True,
            #momentum=self._OPTIMIZER_MOMENTUM
        ),
        run_eagerly=True
    )


#def resblock(x, filters, strides):
#    if strides == 1:
#        x_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2))(x)
#        x_conv = tf.keras.layers.BatchNormalization()(x_conv)
#        fx = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(2, 2))(x)
#    else:
#        x_conv = x
#        fx = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
#    fx = tf.keras.layers.BatchNormalization()(fx)
#    fx = tf.keras.layers.Activation("relu")(fx)
#    fx = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(fx)
#    out = tf.keras.layers.Add()([x_conv, fx])
#    out = tf.keras.layers.BatchNormalization()(out)
#    out = tf.keras.layers.ReLU()(out)
#    return out


if __name__ == '__main__':
    ## Defining the different Models to trtain
    models = ['regressorMSE', 'classificatorXE', 'classificatorEMD']

    userInput = builtins.input(
        'Please enter which Model you want to train:\n 1:regressorMSE \n 2:classificatorXE \n 3:classificatorEMD \n 4:End Training \n')
    while userInput != 4:
        match userInput:
            case "1":
                checkpoint_path = "./checkpoints/regressorMSE/regressorMSE.ckpt"
                checkpoint_path_check = "./checkpoints/regressorMSE/checkpoint"
                db = "imdb"
                image_dir = Path('./{}_crop'.format(db))
                # image_dir = Path('./age_prediction')
                mat_path = './{}_crop/{}.mat'.format(db, db)

                ## Getting the Filepath of the Images and the Labels and concat it in Panda Array(Series) aka Dataframe called images
                filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name="Filepath").astype(str)
                ages = pd.Series(get_age(mat_path, db), name="Age")
                #ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name="Age").astype(int)
                images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).dropna().reset_index(drop=True)

                ## use only 10000 images to speed up training time
                image_df = images.sample(10000, random_state=np.random.randint(1000)).reset_index(drop=True)

                ## splitting images into train and test
                train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)

                ## Defining the ImageDataGenerator and what Preprocessing should be done to the images and normalize each image in image_df to have mean of 0 and deviation of 1
                train_generator = pre_processing.preprocess_TrainImages(train_df)
                test_generator = pre_processing.preprocess_TrainImages(test_df)

                #train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                #    validation_split=0.2,
                #    featurewise_center=True,
                #    featurewise_std_normalization=True
                #)

                #test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                #    featurewise_center=True,
                #    featurewise_std_normalization=True
                #)

                #imgs = []
                #for i in train_df["Filepath"]:
                #    image = tf.keras.preprocessing.image.img_to_array(keras.utils.load_img(i, target_size=(244, 244)))
                #    imgs.append(image)
                #imgs = np.stack(imgs)
                #print(imgs.shape)
                #train_generator.fit(imgs)

                #imgs = []
                #for i in test_df["Filepath"]:
                #    image = tf.keras.preprocessing.image.img_to_array(ut.load_img(i, target_size=(244, 244)))
                #    imgs.append(image)
                #imgs = np.stack(imgs)
                #print(imgs.shape)
                #test_generator.fit(imgs)

                # Getting the Images from Filepath
                train_images = train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="Filepath",
                    y_col="Age",
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="raw",
                    batch_size=64,
                    shuffle=True,
                    seed=41,
                    subset="training"
                )

                val_images = train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="Filepath",
                    y_col="Age",
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="raw",
                    batch_size=64,
                    shuffle=True,
                    seed=41,
                    subset="validation"
                )

                test_images = test_generator.flow_from_dataframe(
                    dataframe=test_df,
                    x_col="Filepath",
                    y_col="Age",
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="raw",
                    batch_size=64,
                    shuffle=False
                )

                if os.path.exists(checkpoint_path):
                    print("Model detected. Loading Model...")
                    model = keras.models.load_model(checkpoint_path)
                else:
                    print("No Model detected. Bilding new Model...")
                    input = tf.keras.Input(shape=(224, 224, 3))
                    #out0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same")(input)
                    #out1 = resblock(out0, filters=64, strides=0)
                    #out2 = resblock(out1, filters=64, strides=0)
                    #out3 = resblock(out2, filters=128, strides=1)
                    #out4 = resblock(out3, filters=128, strides=0)
                    #out5 = resblock(out4, filters=256, strides=1)
                    #out6 = resblock(out5, filters=256, strides=0)
                    #out7 = resblock(out6, filters=512, strides=1)
                    #out8 = resblock(out7, filters=512, strides=0)
                    #out = tf.keras.layers.GlobalAveragePooling2D()(out8)
                    #out = operations.build_general_model(input, 1)
                    #output = tf.keras.layers.Dense(1, activation="relu")(out)
                    output = operations.build_general_model(input=input, nr_classes=1)
                    model = keras.Model(input, output)

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss="mse"
                )
                #print(model.trainable_weights)
                history = model.fit(
                    x=train_images,
                    validation_data=val_images,
                    epochs=20,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss",
                            patience=10,
                            restore_best_weights=True,
                            mode="min"
                        ),
                        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                           save_weights_only=False,
                                                           mode="min",
                                                           verbose=1,
                                                           save_best_only=True,
                                                           initial_value_threshold=168)
                    ]
                )

                print("Training ended. Now to Testing")

                predicted_ages = model.predict(test_images)
                actual_ages = test_images.labels

                print("Testing finished. Results are: ")

                rmse = np.sqrt(model.evaluate(test_images, verbose=1))
                print("Test RMSE: {:.5f}".format(rmse))

                r2 = r2_score(actual_ages, predicted_ages)
                print("Test R^2 Score: {:.5f}".format(r2))
                print(np.average(predicted_ages))
                print(np.average(actual_ages))

                builtins.input("Press something to continue...")

                userInput = builtins.input(
                    'Training finished. If you want to continue, please enter which Model you want to train:\n 1:regressorMSE \n 2:classificatorXE \n 3:classificatorEMD \n 4:End Program \n')

                continue

            case "2":
                #nr_classes = 30
                nr_classes = int(builtins.input('Please enter the Number of Classes (between 1 and 100):'))
                print("Number of Classes selected as {}".format(nr_classes))
                print("Next Step is preparing the Data. Please Wait...")
                checkpoint_path = "./checkpoints/classificatorXE/{}/classificatorXE.ckpt".format(nr_classes)
                checkpoint_path_check = "./checkpoints/classificatorXE/{}/checkpoint".format(nr_classes)
                db = "imdb"
                image_dir = Path('./{}_crop'.format(db))
                # image_dir = Path('./age_prediction')
                mat_path = './{}_crop/{}.mat'.format(db, db)

                ## Getting the Filepath of the Images and the Labels and concat it in Panda Array(Series) aka Dataframe called images
                filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name="Filepath").astype(str)
                ages = pd.Series(get_age(mat_path, db), name="Age")
                images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).dropna().reset_index(drop=True)

                ## Calculation of the Age Groups depending on the Nr of Classes declared earlier
                max_age = ages.max()
                interval = max_age/nr_classes
                floored_interval = math.floor(interval)
                ceiled_interval = math.ceil(interval)
                i = 0
                classes = []
                while len(classes) < nr_classes:
                    classes.append('{}-{}'.format(i, i + floored_interval-1))
                    #classes[i] = "{}-{}".format(i, i + floored_interval)
                    i += floored_interval
                #classes[(x // interval) * math.floor(interval)]
                images["Age"] = pd.Series(images["Age"].apply(
                    lambda x: "{}-{}".format(int((x // interval) * floored_interval),
                                             int((x // interval) * floored_interval + floored_interval-1))))

                ## use only 10000 images to speed up training time
                image_df = images.sample(20000, random_state=np.random.randint(1000)).reset_index(drop=True)

                ## splitting images into train and test
                train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)

                ## Defining the ImageDataGenerator and what Preprocessing should be done to the images
                ## normalize each image in image_df to have mean of 0 and deviation of 1
                train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                    validation_split=0.2,
                    featurewise_center=True,
                    featurewise_std_normalization=True
                )

                test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                    featurewise_center=True,
                    featurewise_std_normalization=True
                )

                imgs = []
                for i in train_df["Filepath"]:
                    image = tf.keras.preprocessing.image.img_to_array(keras.utils.load_img(i, target_size=(244, 244)))
                    imgs.append(image)
                imgs = np.stack(imgs)
                print(imgs.shape)
                train_generator.fit(imgs)

                imgs = []
                for i in test_df["Filepath"]:
                    image = tf.keras.preprocessing.image.img_to_array(ut.load_img(i, target_size=(244, 244)))
                    imgs.append(image)
                imgs = np.stack(imgs)
                print(imgs.shape)
                test_generator.fit(imgs)

                # Getting the Images from Filepath
                train_images = train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="Filepath",
                    y_col="Age",
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="categorical",
                    batch_size=54,
                    shuffle=True,
                    seed=42,
                    subset="training",
                    classes=classes
                )

                val_images = train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="Filepath",
                    y_col="Age",
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="categorical",
                    batch_size=64,
                    shuffle=True,
                    seed=42,
                    subset="validation",
                    classes=classes
                )

                test_images = test_generator.flow_from_dataframe(
                    dataframe=test_df,
                    x_col="Filepath",
                    y_col="Age",
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="categorical",
                    batch_size=64,
                    shuffle=False,
                    classes=classes
                )


                if os.path.exists(checkpoint_path):
                    print("Model detected. Loading Model...")
                    model = keras.models.load_model(checkpoint_path)
                else:
                    print("No Model detected. Loading New One...")
                    input = tf.keras.Input(shape=(224, 224, 3))
                    #out = tf.keras.layers.Dense(nr_classes, activation="relu")(out)
                    #output = tf.keras.layers.Softmax()(out)
                    output = operations.build_general_model(input=input, nr_classes=nr_classes)
                    model = keras.Model(input, output)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.CategoricalCrossentropy()
                )
                #print(model.trainable_weights)
                history = model.fit(
                    x=train_images,
                    validation_data=val_images,
                    epochs=10,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss",
                            patience=5,
                            restore_best_weights=True,
                            mode="min"
                        ),
                        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                           save_weights_only=False,
                                                           mode="min",
                                                           verbose=1,
                                                           save_best_only=True,
                                                           initial_value_threshold=2.925)
                    ]
                )
                print("Training ended. Now to Testing")

                predicted_ages = model.predict(test_images)
                actual_ages = test_images.labels

                print("Testing finished. Results are: ")

                rmse = np.sqrt(model.evaluate(test_images, verbose=1))
                print("Test RMSE: {:.5f}".format(rmse))

                #r2 = r2_score(actual_ages, predicted_ages)
                #print("Test R^2 Score: {:.5f}".format(r2))
                print(np.average(predicted_ages))
                print(np.average(actual_ages))


                rec_score = recall_score(actual_ages, predicted_ages)
                print("Test Recall Score: {:.5f}".format(rec_score))

                builtins.input("Press something to continue...")

                userInput = builtins.input(
                    'Training finished. If you want to continue, please enter which Model you want to train:\n 1:regressorMSE \n 2:classificatorXE \n 3:classificatorEMD \n 4:End Program \n')

            case "3":
                #nr_classes = 30
                nr_classes = int(builtins.input('Please enter the Number of Classes (between 1 and 100):'))
                print("Number of Classes selected as {}".format(nr_classes))
                print("Next Step is preparing the Data. Please Wait...")
                checkpoint_path = "./checkpoints/classificatorEMD/{}/classificatorEMD.ckpt".format(nr_classes)
                checkpoint_path_check = "./checkpoints/classificatorEMD/{}/checkpoint".format(nr_classes)
                db = "imdb"
                image_dir = Path('./{}_crop'.format(db))
                # image_dir = Path('./age_prediction')
                mat_path = './{}_crop/{}.mat'.format(db, db)

                ## Getting the Filepath of the Images and the Labels and concat it in Panda Array(Series) aka Dataframe called images
                filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name="Filepath").astype(str)
                ages = pd.Series(get_age(mat_path, db), name="Age")
                images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).dropna().reset_index(
                    drop=True)

                ## Calculation of the Age Groups depending on the Nr of Classes declared earlier
                max_age = ages.max()
                interval = max_age / nr_classes
                floored_interval = math.floor(interval)
                ceiled_interval = math.ceil(interval)
                i = 0
                classes = []
                while len(classes) < nr_classes:
                    classes.append('{}-{}'.format(i, i + floored_interval - 1))
                    # classes[i] = "{}-{}".format(i, i + floored_interval)
                    i += floored_interval
                # classes[(x // interval) * math.floor(interval)]
                images["Age"] = pd.Series(images["Age"].apply(
                    lambda x: "{}-{}".format(int((x // interval) * floored_interval),
                                             int((x // interval) * floored_interval + floored_interval - 1))))


                ## use only 10000 images to speed up training time and splitting images dataframe into train and test
                image_df = images.sample(10000, random_state=np.random.randint(1000)).reset_index(drop=True)
                train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)


                ## Defining the ImageDataGenerator and what Preprocessing should be done to the images and normalize each image in image_df to have mean of 0 and deviation of 1
                train_generator = pre_processing.preprocess_TrainImages(train_df)
                test_generator = pre_processing.preprocess_TrainImages(test_df)


                # Getting the Images from Filepath
                train_images = train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="Filepath",
                    y_col="Age",
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="categorical",
                    batch_size=54,
                    shuffle=True,
                    seed=42,
                    subset="training",
                    classes=classes
                )

                val_images = train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="Filepath",
                    y_col="Age",
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="categorical",
                    batch_size=64,
                    shuffle=True,
                    seed=42,
                    subset="validation",
                    classes=classes
                )

                test_images = test_generator.flow_from_dataframe(
                    dataframe=test_df,
                    x_col="Filepath",
                    y_col="Age",
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="categorical",
                    batch_size=64,
                    shuffle=False,
                    classes=classes
                )

                if os.path.exists(checkpoint_path):
                    print("Model detected. Loading Model...")
                    model = keras.models.load_model(checkpoint_path)
                else:
                    print("No Model detected. Loading New One...")
                    input = tf.keras.Input(shape=(224, 224, 3))
                    #out = tf.keras.layers.Dense(nr_classes, activation="relu")(out)
                    #output = tf.keras.layers.Softmax()(out)
                    output = operations.build_general_model(input=input, nr_classes=nr_classes)
                    model = keras.Model(input, output)
                #model.compile(
                #    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                #    loss=earth_mover_distance()(model)
                #)
                _compile_model(model=model,loss_function=earth_mover_distance,ground_distance_path=Path('./ground_matrix'))
                #print(model.trainable_weights)
                history = model.fit(
                    x=train_images,
                    validation_data=val_images,
                    epochs=10,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_loss",
                            patience=5,
                            restore_best_weights=True,
                            mode="min"
                        ),
                        tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                           save_weights_only=False,
                                                           mode="min",
                                                           verbose=1,
                                                           save_best_only=True,
                                                           initial_value_threshold=0.071
                                                           ),
                        #GroundDistanceManager(file_path=Path('./ground_matrix'))
                    ]
                )
                print("Training ended. Now to Testing")
                predicted_ages = model.predict(test_images)
                actual_ages = test_images.labels
                print("Testing finished. Results are: ")
                rmse = np.sqrt(model.evaluate(test_images, verbose=1))
                print("Test RMSE: {:.5f}".format(rmse))

                #r2 = r2_score(actual_ages, predicted_ages)
                #print("Test R^2 Score: {:.5f}".format(r2))


                print(np.average(predicted_ages))
                print(np.average(actual_ages))

                #print("Recall Score:")
                #recall_score = recall_score(actual_ages, predicted_ages)

                builtins.input("Press something to continue...")

                userInput = builtins.input(
                    'Training finished. If you want to continue, please enter which Model you want to train:\n 1:regressorMSE \n 2:classificatorXE \n 3:classificatorEMD \n 4:End Program \n')

                continue

            case "4":
                break
            case _:
                print("No Model selected. Please try again.")
                userInput = builtins.input(
                    'Please enter which Model you want to train:\n 1:regressorMSE \n 2:classificatorXE \n 3:classificatorEMD \n 4:End Training  \n')
                continue

    print("Program ended")
    exit()


