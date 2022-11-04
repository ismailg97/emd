import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import os.path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import keras
import ktrain
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL.Image as pil


from models.model_from_paper import PaperModel
from models.standard_model import MyModel


def resblock( x, filters, strides):
    if (strides == 1):
        x_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2))(x)
        x_conv = tf.keras.layers.BatchNormalization()(x_conv)
        # x_conv = x
        fx = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(2, 2))(x)
        # fx = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    else:
        # x_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1))(x)
        # x_conv = tf.keras.layers.BatchNormalization()(x_conv)
        x_conv = x
        fx = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)

    fx = tf.keras.layers.BatchNormalization()(fx)
    fx = tf.keras.layers.Activation("relu")(fx)
    fx = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(fx)
    out = tf.keras.layers.Add()([x_conv, fx])
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.ReLU()(out)
    #print(x_conv.shape)
    #print(fx.shape)
    return out



if __name__ == '__main__':
    #Getting the Filepath of the Images and the Labels and concat it in Panda Array(Series) aka Dataframe called images
    image_dir = Path('./imdb_crop')
    #image_dir = Path('./age_prediction')
    #filepaths = getFilePaths(image_dir)
    filepaths = pd.Series(list(image_dir.glob(r'**/*.jpg')), name="Filepath").astype(str)
    ##show Image
    #print(filepaths.values[0])
    #img = pil.open(filepaths.values[0])
    #img.show()
    #exit()
    ages = pd.Series(filepaths.apply(lambda x: os.path.split(os.path.split(x)[0])[1]), name="Age").astype(int)
    images = pd.concat([filepaths, ages], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)



    #use only 10000 images to speed up training time
    image_df = images.sample(10000, random_state=1).reset_index(drop=True)
    #splitting images into train and test
    train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)
    #print(train_df)


    # Defining the ImageDataGenerator and what Preprocessing should be done to the images
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
    )

    # Getting the Images from Filepath
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col="Filepath",
        y_col="Age",
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
        y_col="Age",
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


    #img1 = train_df['Filepath'].iloc[0]
    #img = pil.open(img1)
    #img.show()
    #print(img.size)
    #print()
    #exit()



    #model = MyModel()

    input = tf.keras.Input(shape=(224,224,3))
    out0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same")(input)
    out1 = resblock(out0, filters=64, strides=0)
    out2 = resblock(out1, filters=64, strides=0)
    out3 = resblock(out2, filters=128, strides=1)
    out4 = resblock(out3, filters=128, strides=0)
    out5 = resblock(out4, filters=256, strides=1)
    out6 = resblock(out5, filters=256, strides=0)
    out7 = resblock(out6, filters=512, strides=1)
    out8 = resblock(out7, filters=512, strides=0)
    out = tf.keras.layers.GlobalAveragePooling2D()(out8)
    #print(self.dense(out).shape)
    output = tf.keras.layers.Dense(1, activation="relu")(out)

    regressor = keras.Model(input, output, name="regressor")
    regressor.summary()
    #exit()

    model = regressor

    #model = PaperModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse"
    )
    history = model.fit(
        x=train_images,
        validation_data=val_images,
        epochs=30,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
        ]
    )

    predicted_ages = model.predict(test_images)
    actual_ages = test_images.labels

    rmse = np.sqrt(model.evaluate(test_images, verbose=0))
    print("Test RMSE: {:.5f}".format(rmse))

    r2 = r2_score(actual_ages, predicted_ages)
    print("Test R^2 Score: {:.5f}".format(r2))
    print(np.average(predicted_ages))
    print(np.average(actual_ages))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
