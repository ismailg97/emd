import builtins
import math
import os.path
from pathlib import Path


import tensorflow as tf
import keras
import keras.utils as ut
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, recall_score
from sklearn.model_selection import train_test_split

from data_handlers import pre_processing
from loss_functions.emd import earth_mover_distance, self_guided_earth_mover_distance, \
    GroundDistanceManager, EmdWeightHeadStart
from models import operations
from src.utils import get_age
from data_handlers.generators import getDataRegression, getDataClassification

if __name__ == '__main__':

    ## Defining the different Models to trtain
    models = ['regressorMSE', 'classificatorXE', 'classificatorEMD']

    userInput = builtins.input(
        'Please enter which Model you want to train:\n 1:regressorMSE \n 2:classificatorXE \n 3:classificatorEMD \n 4:classificatorXEMD \n 5:End Training \n')
    while userInput != 5:
        match userInput:
            case "1":
                print("Next Step is preparing the Data. Please Wait...")
                checkpoint_path = "./checkpoints/regressorMSE/regressorMSE.ckpt"
                image_df = getDataRegression(db="imdb")

                ## splitting images into train and test
                train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)

                ## Defining the ImageDataGenerator and what Preprocessing should be done to the images and normalize each image in image_df to have mean of 0 and deviation of 1
                train_generator = pre_processing.preprocess_TrainImages(train_df)
                test_generator = pre_processing.preprocess_TestImages(test_df)

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
                    output = operations.build_general_model(input=input, nr_classes=1)
                    model = keras.Model(input, output)

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss="mse"
                )
                # print(model.trainable_weights)
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

                age_comparison = zip(predicted_ages, actual_ages)

                print(age_comparison)


                rmse = np.sqrt(model.evaluate(test_images, verbose=1))
                print("Test RMSE: {:.5f}".format(rmse)) #12.8

                r2 = r2_score(actual_ages, predicted_ages)
                print("Test R^2 Score: {:.5f}".format(r2))
                #print(np.average(predicted_ages))
                #print(np.average(actual_ages))

                builtins.input("Press something to continue...")

                userInput = builtins.input(
                    'Training finished. If you want to continue, please enter which Model you want to train:\n 1:regressorMSE \n 2:classificatorXE \n 3:classificatorEMD \n 4:classificatorXEMD \n 5:End Program \n')
                continue

            case "2":
                # nr_classes = 30
                nr_classes = int(builtins.input('Please enter the Number of Classes (between 1 and 100):'))
                print("Number of Classes selected as {}".format(nr_classes))
                print("Next Step is preparing the Data. Please Wait...")
                checkpoint_path = "./checkpoints/classificatorXE/{}/classificatorXE.ckpt".format(nr_classes)
                #db = "imdb"
                #image_dir = Path('./{}_crop'.format(db))
                ## image_dir = Path('./age_prediction')
                #mat_path = './{}_crop/{}.mat'.format(db, db)

                ## Getting the Filepath of the Images and the Labels and concat it in Panda Array(Series) aka Dataframe called images
                #filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name="Filepath").astype(str)
                #ages = pd.Series(get_age(mat_path, db), name="Age")
                #images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).dropna().reset_index(
                #    drop=True)

                ## Calculation of the Age Groups depending on the Nr of Classes declared earlier
                #max_age = ages.max()
                #interval = max_age / nr_classes
                #floored_interval = math.floor(interval)
                #ceiled_interval = math.ceil(interval)
                #i = 0
                #classes = []
                #while len(classes) < nr_classes:
                #    classes.append('{}-{}'.format(i, i + floored_interval - 1))
                #    i += floored_interval
                #images["Age"] = pd.Series(images["Age"].apply(
                #    lambda x: "{}-{}".format(int((x // interval) * floored_interval),
                #                             int((x // interval) * floored_interval + floored_interval - 1))))

                ## use only 10000 images to speed up training time
                #image_df = images.sample(20000, random_state=np.random.randint(1000)).reset_index(drop=True)
                image_df, classes = getDataClassification(db="imdb", nr_classes=nr_classes)

                ## splitting images into train and test
                train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)

                ## Defining the ImageDataGenerator and what Preprocessing should be done to the images and normalize each image in image_df to have mean of 0 and deviation of 1
                train_generator = pre_processing.preprocess_TrainImages(train_df)
                test_generator = pre_processing.preprocess_TestImages(test_df)

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
                    output = operations.build_general_model(input=input, nr_classes=nr_classes)
                    model = keras.Model(input, output)
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.CategoricalCrossentropy()
                )
                history = model.fit(
                    x=train_images,
                    validation_data=val_images,
                    epochs=20,
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
                                                           #initial_value_threshold=2.925 #30
                        )
                    ]
                )
                print("Training ended. Now to Testing")

                predicted_ages = model.predict(test_images)
                actual_ages = test_images.labels

                print("Testing finished. Results are: ")

                #rmse = np.sqrt(model.evaluate(test_images, verbose=1))
                #print("Test RMSE: {:.5f}".format(rmse))

                age_comparison = zip(predicted_ages,actual_ages)

                print(age_comparison)

                #print(np.average(predicted_ages))
                #print(np.average(actual_ages))

                rec_score = recall_score(actual_ages, predicted_ages)
                print("Test Recall Score: {:.5f}".format(rec_score))

                builtins.input("Press something to continue...")

                userInput = builtins.input(
                    'Training finished. If you want to continue, please enter which Model you want to train:\n 1:regressorMSE \n 2:classificatorXE \n 3:classificatorEMD \n 4:classificatorXEMD \n 5:End Program \n')
            case "3":
                nr_classes = int(builtins.input('Please enter the Number of Classes (between 1 and 100):'))
                print("Number of Classes selected as {}".format(nr_classes))
                print("Next Step is preparing the Data. Please Wait...")
                checkpoint_path = "./checkpoints/classificatorEMD/{}/classificatorEMD.ckpt".format(nr_classes)
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
                    i += floored_interval
                images["Age"] = pd.Series(images["Age"].apply(
                    lambda x: "{}-{}".format(int((x // interval) * floored_interval),
                                             int((x // interval) * floored_interval + floored_interval - 1))))

                # while len(classes) < nr_classes:
                #    classes.append(i)
                #    i += floored_interval
                # images["Age"] = pd.Series(images["Age"].apply(lambda y: min(classes, key=lambda x: abs(x-y)) ))
                # print(classes)
                # print(images)
                # exit()

                ## use only 10000 images to speed up training time and splitting images dataframe into train and test
                image_df = images.sample(10000, random_state=np.random.randint(1000)).reset_index(drop=True)
                train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)

                ## Defining the ImageDataGenerator and what Preprocessing should be done to the images and normalize each image in image_df to have mean of 0 and deviation of 1
                train_generator = pre_processing.preprocess_TrainImages(train_df)
                test_generator = pre_processing.preprocess_TestImages(test_df)

                # print(test_generator)

                # Getting the Images from Filepath
                train_images = train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="Filepath",
                    y_col="Age",
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="categorical",
                    # class_mode="raw",
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
                    # class_mode="raw",
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
                    # class_mode="raw",
                    batch_size=64,
                    shuffle=False,
                    classes=classes
                )

                print(train_images.labels)

                if os.path.exists(checkpoint_path):
                    print("Model detected. Loading Model...")
                    model = keras.models.load_model(checkpoint_path, compile=False)
                else:
                    print("No Model detected. Loading New One...")
                    input = tf.keras.Input(shape=(224, 224, 3))
                    output = operations.build_general_model(input=input, nr_classes=nr_classes)
                    model = keras.Model(input, output)

                # model.summary()
                # exit()

                # loss_function = self_guided_earth_mover_distance
                # _compile_model(model=model, loss_function=loss_function)
                model.compile(
                    loss=earth_mover_distance(model=model),
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=0.001
                    ),
                    metrics="accuracy"
                )

                history = model.fit(
                    x=train_images,
                    validation_data=val_images,
                    epochs=20,
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=5,
                        restore_best_weights=True,
                        mode="min"
                    ),
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=checkpoint_path,
                            save_weights_only=False,
                            mode="min",
                            verbose=1,
                            save_best_only=True,
                            #initial_value_threshold=0.0736 #30
                            initial_value_threshold=0.07287 #60

                        )
                    ]
                )

                print("Training ended. Now to Testing")
                predicted_ages = np.asarray(model.predict(test_images))
                print(np.argmax(predicted_ages, axis=-1))
                # predicted_ages = np.argmax(predicted_ages)
                actual_ages = np.asarray(tf.one_hot(indices=test_images.labels, depth=nr_classes))
                # print(predicted_ages)
                # print(actual_ages)

                print("Testing finished. Results are: ")
                rmse = np.sqrt(model.evaluate(test_images, verbose=1))
                print("Test RMSE: {:.5f}".format(rmse))

                print("Average of Predicted Ages: {}".format(np.average(predicted_ages)))
                print("Average of Actual Ages: {}".format(np.average(actual_ages)))

                rc_score = recall_score(actual_ages.argmax(axis=1), predicted_ages.argmax(axis=1), average="micro")
                print("Recall Score: {}".format(rc_score))

                builtins.input("Press something to continue...")

                userInput = builtins.input(
                    'Training finished. If you want to continue, please enter which Model you want to train:\n 1:regressorMSE \n 2:classificatorXE \n 3:classificatorEMD \n 4:classificatorXEMD \n 5:End Program \n')
                continue
            case "4":
                nr_classes = int(builtins.input('Please enter the Number of Classes (between 1 and 100):'))
                print("Number of Classes selected as {}".format(nr_classes))
                print("Next Step is preparing the Data. Please Wait...")
                checkpoint_path = "./checkpoints/classificatorEMD/{}/classificatorXEMD.ckpt".format(nr_classes)
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
                    i += floored_interval
                images["Age"] = pd.Series(images["Age"].apply(
                    lambda x: "{}-{}".format(int((x // interval) * floored_interval),
                                             int((x // interval) * floored_interval + floored_interval - 1))))

                ## use only 10000 images to speed up training time and splitting images dataframe into train and test
                image_df = images.sample(100, random_state=np.random.randint(1000)).reset_index(drop=True)
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
                    batch_size=32,
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
                    batch_size=32,
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
                    batch_size=32,
                    shuffle=False,
                    classes=classes
                )

                # print(train_images.labels)
                # exit()

                if os.path.exists(checkpoint_path):
                    print("Model detected. Loading Model...")
                    model = keras.models.load_model(checkpoint_path, compile=False)
                else:
                    print("No Model detected. Loading New One...")
                    input = tf.keras.Input(shape=(224, 224, 3))
                    output = operations.build_general_model(input=input, nr_classes=nr_classes)
                    model = tf.keras.Model(input, output)

                    # submodel = keras.Model(input, model.get_layer(index=-2).output)
                    # results = submodel.predict(train_images)
                    # print(model.get_layer(index=-2).summary())
                    # exit()
                    # intermediate_layer_model = keras.Model(input,
                    #                                       model.get_layer(index=-2).output)
                    # intermediate_output = intermediate_layer_model.predict(x=test_images, batch_size=1)
                    # print(intermediate_output.shape)
                    # print(model.layers[-2].output)
                    # exit()
                    output = operations.get_second_layer(input=input, nr_classes=nr_classes)
                    # print(output.shape)
                    submodel = tf.keras.Model(input, output)
                    second_to_last_layer_output = submodel.predict(train_images)
                    layer = tf.convert_to_tensor(second_to_last_layer_output)
                    # second_to_last_layer = model.layers[-2](y)
                    # print(layer)
                    # print(layer.shape)
                    # exit()
                    # print(np.argmax(second_to_last_layer_output, axis=1))
                    # exit()

                    # functor = K.function(inputs=input,outputs=output)
                    # print(functor(train_images))
                    # exit()

                    # model.get_layer(index=-2).predict()

                    emd_weight_head_start = EmdWeightHeadStart()
                    ground_distance_manager = GroundDistanceManager(Path('./ground_matrix'))
                    ground_distance_manager.set_labels(tf.one_hot(indices=train_images.labels, depth=nr_classes))
                    setattr(model, 'emd_weight_head_start', emd_weight_head_start)
                    setattr(model, 'ground_distance_manager', ground_distance_manager)
                    # setattr(model, 'second_to_last_layer', model.layers[-2].output)
                    setattr(model, 'second_to_last_layer', submodel.predict(train_images))
                    # setattr(model, 'nr_classes', nr_classes)

                    # model = XEMDModel(emd_weight_head_start=emd_weight_head_start,ground_distance_manager=ground_distance_manager,nr_classes=nr_classes)

                    # model.summary()
                    # exit()

                # model = XEMDModel(nr_classes=nr_classes, ground_distance_manager=ground_distance_manager,
                #                  emd_weight_head_start=emd_weight_head_start)
                # callbacks = getCallbacks(model, loss_function, checkpoint_path=checkpoint_path, labels=train_images.labels)
                # _compile_model(model=model, loss_function=loss_function, ground_distance_sensitivity=1,ground_distance_bias=0.5)
                loss_function = self_guided_earth_mover_distance
                model.compile(
                    loss=loss_function(model=model, ground_distance_sensitivity=1, ground_distance_bias=0.5),
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=0.001
                    ),
                    run_eagerly=True
                )
                history = model.fit(
                    x=train_images,
                    validation_data=val_images,
                    epochs=10,
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=5,
                        restore_best_weights=True,
                        mode="min"
                    ),
                        tf.keras.callbacks.ModelCheckpoint(
                            filepath=checkpoint_path,
                            save_weights_only=False,
                            mode="min",
                            verbose=1,
                            save_best_only=True,
                            initial_value_threshold=0.075
                        ),
                        model.emd_weight_head_start,
                        model.ground_distance_manager
                    ]
                )

                print("Training ended. Now to Testing")
                predicted_ages = model.predict(test_images)
                actual_ages = test_images.labels
                print("Testing finished. Results are: ")
                rmse = np.sqrt(model.evaluate(test_images, verbose=1))
                print("Test RMSE: {:.5f}".format(rmse))

                print("Average of Predicted Ages: {}".format(np.average(predicted_ages)))
                print("Average of Actual Ages: {}".format(np.average(actual_ages)))

                rc_score = recall_score(actual_ages, predicted_ages)
                print("Recall Score: {}".format(rc_score))
                builtins.input("Press something to continue...")

                userInput = builtins.input(
                    'Training finished. If you want to continue, please enter which Model you want to train:\n 1:regressorMSE \n 2:classificatorXE \n 3:classificatorEMD \n 4:classificatorXEMD \n 5:End Program \n')

                continue
            case "5":
                break
            case _:
                print("No Model selected. Please try again.")
                userInput = builtins.input(
                    'Please enter which Model you want to train:\n 1:regressorMSE \n 2:classificatorXE \n 3:classificatorEMD \n 4:classificatorXEMD \n 5:End Program \n')
                continue

    print("Program ended")
    exit()
