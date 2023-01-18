import builtins
import math
import os.path
from pathlib import Path


import tensorflow as tf
import keras
import keras.utils as ut
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, recall_score, precision_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_handlers import pre_processing
from loss_functions.emd import earth_mover_distance, self_guided_earth_mover_distance, \
    GroundDistanceManager, EmdWeightHeadStart
from models import operations
from src.utils import get_age
from data_handlers.generators import getDataRegression, getDataClassification

from matplotlib import pyplot as plt

if __name__ == '__main__':
    ## Defining the different Models to trtain
    models = ['regressorMSE', 'classificatorXE', 'classificatorEMD']

    userInput = builtins.input(
        'Please enter which Model you want to train:\n 1:regressorMSE \n 2:classificatorXE \n 3:classificatorEMD \n 4:classificatorXEMD \n 5:End Training \n')
    while userInput != 5:
        match userInput:
            case "1":
                print("Next Step is preparing the Data. Please Wait...")
                checkpoint_path = "./checkpoints/regressorMSE/wiki/regressorMSE.ckpt"
                image_df = getDataRegression(db="wiki")

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
                    loss="mse",
                    metrics=["accuracy"]
                )


                # print(model.trainable_weights)
                history = model.fit(
                    x=train_images,
                    validation_data=val_images,
                    epochs=30,
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
                                                           #initial_value_threshold=170 #168
                                                           )
                    ]
                )

                print(history.history.keys())
                #  "Accuracy"
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                plt.savefig(fname="./checkpoints/regressorMSE/accuracy_fig")
                plt.show()
                # "Loss"
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                plt.savefig(fname="./checkpoints/regressorMSE/loss_fig")
                plt.show()

                # All
                pd.DataFrame(history.history).plot(figsize=(8, 5))
                plt.savefig(fname="./checkpoints/regressorMSE/all_fig")
                plt.show()


                print("Training ended. Now to Testing")

                test_loss, test_acc = model.evaluate(x=test_images)
                print("Loss: {}".format(test_loss), "Accuracy: {}".format(test_acc))

                predicted_ages = model.predict(test_images)
                actual_ages = test_images.labels

                print("Testing finished. Results are: ")

                rmse = np.sqrt(model.evaluate(test_images, verbose=1))
                print("Test RMSE: {:.5f}".format(rmse)) #12.8

                print(np.average(predicted_ages))
                print(np.average(actual_ages))

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

                ## use only 10000 images to speed up training time
                image_df, max_age = getDataClassification(db="wiki", nr_classes=nr_classes)
                train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)

                test_labels = test_df["Age"]

                interval = max_age / nr_classes
                print(interval)
                floored_interval = math.floor(interval)
                i = 0
                classes = []
                while len(classes) < nr_classes:
                    # classes.append(i + 0.5 * interval)
                    # i += interval
                    classes.append(i)
                    i += 1
                train_df["Age"] = pd.Series(train_df["Age"].apply(
                    lambda x: int(x // interval)))

                test_df["Age"] = pd.Series(test_df["Age"].apply(
                    lambda x: int(x // interval)))

                # Get one hot encoding of columns B
                one_hot = pd.get_dummies(train_df['Age'])
                #print(one_hot)
                # Drop column B as it is now encoded
                train_df = train_df.drop('Age', axis=1)
                # Join the encoded df
                train_df = train_df.join(one_hot)

                # Get one hot encoding of columns B
                one_hot = pd.get_dummies(test_df['Age'])
                #print(one_hot)
                # Drop column B as it is now encoded
                test_df = test_df.drop('Age', axis=1)
                # Join the encoded df
                test_df = test_df.join(one_hot)


                for column in classes:
                    if column not in train_df.columns:
                        # If the element is not present, create a new column with all values set to 0.
                        train_df[column] = 0
                    if column not in test_df.columns:
                        test_df[column] = 0



                columns = ["Filepath"]
                for col in classes:
                    columns.append(col)
                train_df = train_df[columns]
                test_df = test_df[columns]


                print(train_df)

                print(test_df)


                ## Defining the ImageDataGenerator and what Preprocessing should be done to the images and normalize each image in image_df to have mean of 0 and deviation of 1
                train_generator = pre_processing.preprocess_TrainImages(train_df)
                test_generator = pre_processing.preprocess_TestImages(test_df)

                # Getting the Images from Filepath
                train_images = train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="Filepath",
                    y_col=classes,
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="raw",
                    batch_size=32,
                    shuffle=True,
                    seed=42,
                    subset="training"
                )

                val_images = train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="Filepath",
                    y_col=classes,
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="raw",
                    batch_size=32,
                    shuffle=True,
                    seed=42,
                    subset="validation"
                )

                #test_images = test_generator.flow_from_dataframe(
                #    dataframe=test_df,
                #    x_col="Filepath",
                #    y_col="Age",
                #    target_size=(224, 224),
                #    color_mode="rgb",
                #    class_mode="raw",
                #    batch_size=64,
                #    shuffle=False
                #)

                test_images = test_generator.flow_from_dataframe(
                    dataframe=test_df,
                    x_col="Filepath",
                    y_col=classes,
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="raw",
                    batch_size=32,
                    shuffle=False
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
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=["accuracy"]
                )

                #print(model.summary())

                history = model.fit(
                    x=train_images,
                    validation_data=val_images,
                    epochs=50,
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
                                                           #initial_value_threshold=1.83298 #10
                                                           #initial_value_threshold=3.05373 #30
                                                           #initial_value_threshold=3.82589 #60

                        )
                    ]
                )

                print(history.history.keys())
                #  "Accuracy"
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.title('model accuracy - XE - {} Classs'.format(nr_classes))
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                plt.savefig(fname="./checkpoints/classificatorXE/{}/accuracy_fig-4".format(nr_classes))
                plt.show()
                # "Loss"
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss - XE - {} Classes'.format(nr_classes))
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                plt.savefig(fname="./checkpoints/classificatorXE/{}/loss_fig-4".format(nr_classes))
                plt.show()

                # All
                pd.DataFrame(history.history).plot(figsize=(8, 5))
                plt.savefig(fname="./checkpoints/classificatorXE/{}/all_fig-4".format(nr_classes))
                plt.show()

                print("Training ended. Now to Testing")

                print("Real Labels: {}".format(test_labels))

                classified_labels = test_images.labels
                print("Classified Labels: {}".format(classified_labels))

                labels_index = tf.argmax(classified_labels, 1)
                print(labels_index)

                test_loss, test_acc = model.evaluate(x=test_images)
                print("Loss: {}".format(test_loss), "Accuracy: {}".format(test_acc))

                classified_prediction = model.predict(test_images)
                print("Model Prediction: {}".format(classified_prediction))

                prediction_index = tf.argmax(classified_prediction, 1)
                print(prediction_index)

                prediction = tf.argmax(classified_prediction, 1) * interval + 0.5 * interval
                print(prediction)

                avg_predicted_ages = []
                for arr in classified_prediction:
                    sum_value = 0
                    for i, prob in enumerate(arr):
                        sum_value += (i * interval + interval * 0.5) * prob
                    avg_predicted_ages.append(sum_value)
                #print(avg_predicted_ages)

                m = tf.keras.metrics.RootMeanSquaredError()
                m.update_state(test_labels, avg_predicted_ages)
                keren_result = m.result().numpy()

                # actual_actual_ages = tf.argmax(classified_actual_ages, 1) * interval + 0.5 * interval
                declassified_labels = labels_index * interval + 0.5 * interval
                print(declassified_labels)

                m = tf.keras.metrics.RootMeanSquaredError()
                m.update_state(test_labels, prediction)
                result = m.result().numpy()

                RMSE_keren = mean_squared_error(test_labels, avg_predicted_ages) ** 0.5

                RMSE_class = mean_squared_error(declassified_labels, prediction) ** 0.5

                RMSE_cont = mean_squared_error(test_labels, prediction) ** 0.5

                print("Testing finished. Results are: ")

                print("Test Keren RMSE: {:.5f}".format(keren_result))  # rsme10 #rsme30  #rsme60
                print("Test RMSE (Continuos): {:.5f}".format(result))  # rsme10 20.06264 #rsme30  #rsme60
                print("Test RMSE Classified: {:.5f}".format(RMSE_class))  # rsme10  #rsme30  #rsme60

                print("Avg Predicted Ages: {}".format(np.average(prediction)))
                print("Avg Actual Ages: {}".format(np.average(test_labels)))

                OneOff = 0
                for i in range(len(prediction_index)):
                    if math.sqrt((prediction_index[i] - labels_index[i]) ** 2) < 2:
                        OneOff += 1
                OneOffAcc = OneOff / len(prediction_index)
                print("OneOffAcc: {}".format(OneOffAcc))

                rec_score = recall_score(labels_index, predicted_index, average="macro")
                print("Test Recall Score: {}".format(rec_score))

                precision_sc = precision_score(labels_index, prediction_index, average='macro')
                print("Test Precision Score: {}".format(precision_sc))

                exit()


                builtins.input("Press something to continue...")

                userInput = builtins.input(
                    'Training finished. If you want to continue, please enter which Model you want to train:\n 1:regressorMSE \n 2:classificatorXE \n 3:classificatorEMD \n 4:classificatorXEMD \n 5:End Program \n')
            case "3":
                nr_classes = int(builtins.input('Please enter the Number of Classes (between 1 and 100):'))
                print("Number of Classes selected as {}".format(nr_classes))
                print("Next Step is preparing the Data. Please Wait...")
                checkpoint_path = "./checkpoints/classificatorEMD/{}/classificatorEMD.ckpt".format(nr_classes)


                ## use only 10000 images to speed up training time and splitting images dataframe into train and test
                #image_df = images.sample(10000, random_state=np.random.randint(1000)).reset_index(drop=True)
                image_df, max_age = getDataClassification(db="imdb", nr_classes=nr_classes)
                train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)

                test_labels = test_df["Age"]

                interval = max_age / nr_classes
                print(interval)
                floored_interval = math.floor(interval)
                i = 0
                classes = []
                while len(classes) < nr_classes:
                    # classes.append(i + 0.5 * interval)
                    # i += interval
                    classes.append(i)
                    i += 1
                train_df["Age"] = pd.Series(train_df["Age"].apply(
                    lambda x: int(x // interval)))

                test_df["Age"] = pd.Series(test_df["Age"].apply(
                    lambda x: int(x // interval)))

                # Get one hot encoding of columns B
                one_hot = pd.get_dummies(train_df['Age'])
                # print(one_hot)
                # Drop column B as it is now encoded
                train_df = train_df.drop('Age', axis=1)
                # Join the encoded df
                train_df = train_df.join(one_hot)

                # Get one hot encoding of columns B
                one_hot = pd.get_dummies(test_df['Age'])
                # print(one_hot)
                # Drop column B as it is now encoded
                test_df = test_df.drop('Age', axis=1)
                # Join the encoded df
                test_df = test_df.join(one_hot)

                for column in classes:
                    if column not in train_df.columns:
                        # If the element is not present, create a new column with all values set to 0.
                        train_df[column] = 0
                    if column not in test_df.columns:
                        test_df[column] = 0

                columns = ["Filepath"]
                for col in classes:
                    columns.append(col)
                train_df = train_df[columns]
                test_df = test_df[columns]

                #print(train_df)

                #print(test_df)

                ## Defining the ImageDataGenerator and what Preprocessing should be done to the images and normalize each image in image_df to have mean of 0 and deviation of 1
                train_generator = pre_processing.preprocess_TrainImages(train_df)
                test_generator = pre_processing.preprocess_TestImages(test_df)

                # print(test_generator)

                # Getting the Images from Filepath
                train_images = train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="Filepath",
                    y_col=classes,
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="raw",
                    batch_size=32,
                    shuffle=True,
                    seed=42,
                    subset="training"
                )

                val_images = train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="Filepath",
                    y_col=classes,
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="raw",
                    batch_size=32,
                    shuffle=True,
                    seed=42,
                    subset="validation"
                )

                #test_images = test_generator.flow_from_dataframe(
                #    dataframe=test_df,
                #    x_col="Filepath",
                #    y_col="Age",
                #    target_size=(224, 224),
                #    color_mode="rgb",
                #    class_mode="raw",
                #    batch_size=64,
                #    shuffle=False
                #)

                test_images = test_generator.flow_from_dataframe(
                    dataframe=test_df,
                    x_col="Filepath",
                    y_col=classes,
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="raw",
                    batch_size=32,
                    shuffle=False
                )

                #print(train_images.labels)

                if os.path.exists(checkpoint_path):
                    print("Model detected. Loading Model...")
                    model = keras.models.load_model(checkpoint_path, compile=False)
                else:
                    print("No Model detected. Loading New One...")
                    input = tf.keras.Input(shape=(224, 224, 3))
                    output = operations.build_general_model(input=input, nr_classes=nr_classes)
                    model = keras.Model(input, output)

                model.compile(
                    loss=earth_mover_distance(model=model),
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=0.001
                    ),
                    metrics="accuracy"
                )

                print(model.summary())

                history = model.fit(
                    x=train_images,
                    validation_data=val_images,
                    epochs=30,
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
                            #initial_value_threshold=0.09003 #10
                            #initial_value_threshold=0.09125 #30
                            #initial_value_threshold=0.095 #60

                        )
                    ]
                )

                print(history.history.keys())
                #  "Accuracy"
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'])
                plt.title('model accuracy - EMD - {} Classes'.format(nr_classes))
                plt.ylabel('accuracy')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                plt.savefig(fname="./checkpoints/classificatorEMD/{}/accuracy_fig".format(nr_classes))
                plt.show()
                # "Loss"
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss - EMD - {} Classes'.format(nr_classes))
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'val'], loc='upper left')
                plt.savefig(fname="./checkpoints/classificatorEMD/{}/loss_fig".format(nr_classes))
                plt.show()

                # All
                pd.DataFrame(history.history).plot(figsize=(8, 5))
                plt.savefig(fname="./checkpoints/classificatorEMD/{}/all_fig".format(nr_classes))
                plt.show()

                print("Training ended. Now to Testing")

                print("Real Labels: {}".format(test_labels))

                classified_labels = test_images.labels
                print("Classified Labels: {}".format(classified_labels))

                labels_index = tf.argmax(classified_labels, 1)
                print(labels_index)

                declassified_labels = labels_index * interval + 0.5 * interval
                print(declassified_labels)

                test_loss, test_acc = model.evaluate(x=test_images)
                print("Loss: {}".format(test_loss), "Accuracy: {}".format(test_acc))

                classified_prediction = model.predict(test_images)
                print("Model Prediction: {}".format(classified_prediction))

                prediction_index = tf.argmax(classified_prediction, 1)
                print(prediction_index)

                declassified_prediction = prediction_index * interval + 0.5 * interval
                print(declassified_prediction)

                avg_predicted_ages = []
                for arr in classified_prediction:
                    sum_value = 0
                    for i, prob in enumerate(arr):
                        sum_value += (i * interval + interval * 0.5) * prob
                    avg_predicted_ages.append(sum_value)
                #print(avg_predicted_ages)

                m = tf.keras.metrics.RootMeanSquaredError()
                m.update_state(test_labels, declassified_prediction)
                result = m.result().numpy()

                RMSE_keren = mean_squared_error(test_labels, avg_predicted_ages) ** 0.5

                RMSE_class = mean_squared_error(declassified_labels, declassified_prediction) ** 0.5

                OneOff = 0
                for i in range(len(prediction_index)):
                    if math.sqrt((prediction_index[i] - labels_index[i]) ** 2) < 2:
                        OneOff += 1
                OneOffAcc = OneOff / len(prediction_index)

                print("Testing finished. Results are: ")

                print("Test RMSE Continuos: {:.5f}".format(result))  # rsme10 20.06264 #rsme30  #rsme60
                print("Test SKLearn RMSE Classified: {:.5f}".format(RMSE_class))  # rsme10  #rsme30  #rsme60
                print("Test SKLearn RMSE Keren: {:.5f}".format(RMSE_keren))  # rsme10  #rsme30  #rsme60

                print("Average Predicitions: {}".format(np.average(declassified_prediction)))
                print("Average Labels: {}".format(np.average(test_labels)))
                print("Average Classified Labels: {}".format(np.average(classified_labels)))

                print("OneOffAcc: {}".format(OneOffAcc))

                # rec_score = recall_score(, predicted_ages_index, average="macro")
                # print("Test Recall Score: {}".format(rec_score))

                # precision_sc = precision_score(actual_ages_index, predicted_ages_index, average='macro')
                # print("Test Precision Score: {}".format(precision_sc))

                builtins.input("Press something to continue...")

                userInput = builtins.input(
                    'Training finished. If you want to continue, please enter which Model you want to train:\n 1:regressorMSE \n 2:classificatorXE \n 3:classificatorEMD \n 4:classificatorXEMD \n 5:End Program \n')
                continue
            case "4":
                nr_classes = int(builtins.input('Please enter the Number of Classes (between 1 and 100):'))
                print("Number of Classes selected as {}".format(nr_classes))
                print("Next Step is preparing the Data. Please Wait...")
                checkpoint_path = "./checkpoints/classificatorEMD/{}/classificatorXEMD.ckpt".format(nr_classes)

                ## use only 10000 images to speed up training time and splitting images dataframe into train and test
                image_df, max_age = getDataClassification(db="wiki", nr_classes=nr_classes)
                train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)

                interval = max_age / nr_classes
                print(interval)
                floored_interval = math.floor(interval)
                i = 0
                classes = []
                while len(classes) < nr_classes:
                    classes.append(i)
                    i += 1
                train_df["Age"] = pd.Series(train_df["Age"].apply(
                    lambda x: int(x // interval)))

                # Get one hot encoding of columns B
                one_hot = pd.get_dummies(train_df['Age'])
                # Drop column B as it is now encoded
                train_df = train_df.drop('Age', axis=1)
                # Join the encoded df
                train_df = train_df.join(one_hot)

                print(classes)
                print(train_df)
                # exit()

                ## Defining the ImageDataGenerator and what Preprocessing should be done to the images and normalize each image in image_df to have mean of 0 and deviation of 1
                train_generator = pre_processing.preprocess_TrainImages(train_df)
                test_generator = pre_processing.preprocess_TrainImages(test_df)

                # Getting the Images from Filepath
                train_images = train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="Filepath",
                    y_col=classes,
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="raw",
                    batch_size=32,
                    shuffle=True,
                    seed=42,
                    subset="training"
                )

                val_images = train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="Filepath",
                    y_col=classes,
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="raw",
                    batch_size=32,
                    shuffle=True,
                    seed=42,
                    subset="validation"
                )

                test_images = test_generator.flow_from_dataframe(
                    dataframe=test_df,
                    x_col="Filepath",
                    y_col="Age",
                    target_size=(224, 224),
                    color_mode="rgb",
                    class_mode="raw",
                    batch_size=32,
                    shuffle=False
                )

                if os.path.exists(checkpoint_path):
                    print("Model detected. Loading Model...")
                    model = keras.models.load_model(checkpoint_path, compile=False)
                else:
                    print("No Model detected. Loading New One...")
                    input = tf.keras.Input(shape=(224, 224, 3))
                    output = operations.build_general_model(input=input, nr_classes=nr_classes)
                    model = tf.keras.Model(input, output)

                    submodel = tf.keras.Model(input, output)
                    #second_to_last_layer_output = submodel.predict(train_images)
                    #layer = tf.convert_to_tensor(second_to_last_layer_output)
                    emd_weight_head_start = EmdWeightHeadStart()
                    ground_distance_manager = GroundDistanceManager(Path('./ground_matrix'))
                    ground_distance_manager.set_labels(tf.one_hot(indices=train_images.labels, depth=nr_classes))
                    setattr(model, 'emd_weight_head_start', emd_weight_head_start)
                    setattr(model, 'ground_distance_manager', ground_distance_manager)
                    # setattr(model, 'second_to_last_layer', model.layers[-2].output)
                    setattr(model, 'second_to_last_layer', submodel.predict(train_images))

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
