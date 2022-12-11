import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, recall_score
import keras
from data_handlers import pre_processing
from models import operations


from scipy.io import loadmat
from src.utils import get_meta, get_age

if __name__ == '__main__':
    checkpoint_path = "./checkpoints/regressorMSE/regressorMSE.ckpt"
    checkpoint_path_check = "./checkpoints/regressorMSE/checkpoint"
    db = "imdb"
    image_dir = Path('./{}_crop'.format(db))
    # image_dir = Path('./age_prediction')
    mat_path = './{}_crop/{}.mat'.format(db, db)

    ## Getting the Filepath of the Images and the Labels and concat it in Panda Array(Series) aka Dataframe called images
    filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name="Filepath").astype(str)
    ages = pd.Series(get_age(mat_path, db), name="Age")
    # ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name="Age").astype(int)
    images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).dropna().reset_index(drop=True)

    ## use only 10000 images to speed up training time
    image_df = images.sample(10000, random_state=np.random.randint(1000)).reset_index(drop=True)

    ## splitting images into train and test
    train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)

    ## Defining the ImageDataGenerator and what Preprocessing should be done to the images and normalize each image in image_df to have mean of 0 and deviation of 1
    train_generator = pre_processing.preprocess_TrainImages(train_df)
    test_generator = pre_processing.preprocess_TrainImages(test_df)

    # train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    #    validation_split=0.2,
    #    featurewise_center=True,
    #    featurewise_std_normalization=True
    # )

    # test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    #    featurewise_center=True,
    #    featurewise_std_normalization=True
    # )

    # imgs = []
    # for i in train_df["Filepath"]:
    #    image = tf.keras.preprocessing.image.img_to_array(keras.utils.load_img(i, target_size=(244, 244)))
    #    imgs.append(image)
    # imgs = np.stack(imgs)
    # print(imgs.shape)
    # train_generator.fit(imgs)

    # imgs = []
    # for i in test_df["Filepath"]:
    #    image = tf.keras.preprocessing.image.img_to_array(ut.load_img(i, target_size=(244, 244)))
    #    imgs.append(image)
    # imgs = np.stack(imgs)
    # print(imgs.shape)
    # test_generator.fit(imgs)

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
        # out0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same")(input)
        # out1 = resblock(out0, filters=64, strides=0)
        # out2 = resblock(out1, filters=64, strides=0)
        # out3 = resblock(out2, filters=128, strides=1)
        # out4 = resblock(out3, filters=128, strides=0)
        # out5 = resblock(out4, filters=256, strides=1)
        # out6 = resblock(out5, filters=256, strides=0)
        # out7 = resblock(out6, filters=512, strides=1)
        # out8 = resblock(out7, filters=512, strides=0)
        # out = tf.keras.layers.GlobalAveragePooling2D()(out8)
        # out = operations.build_general_model(input, 1)
        # output = tf.keras.layers.Dense(1, activation="relu")(out)
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

    rmse = np.sqrt(model.evaluate(test_images, verbose=1))
    print("Test RMSE: {:.5f}".format(rmse))

    r2 = r2_score(actual_ages, predicted_ages)
    print("Test R^2 Score: {:.5f}".format(r2))
    print(np.average(predicted_ages))
    print(np.average(actual_ages))
    print("Program ended")
    exit()
