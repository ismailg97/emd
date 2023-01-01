import tensorflow as tf
import keras.utils as ut
import numpy as np
import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_TrainImages(train_df):
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        samplewise_center=True,
        samplewise_std_normalization=True,
        validation_split=0.2
    )
    #imgs = []
    #for i in train_df["Filepath"]:
    #    image = tf.keras.preprocessing.image.img_to_array(keras.utils.load_img(i, target_size=(244, 244)))
    #    imgs.append(image)
    #imgs = np.stack(imgs)
    #train_generator.fit(imgs)


    #train_features = np.array(train_df)
    #scaler = StandardScaler()
    #train_features = scaler.fit_transform(train_df)

    return train_generator


def preprocess_TestImages(test_df):
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        #featurewise_center=True,
        #featurewise_std_normalization=True
        samplewise_center=True,
        samplewise_std_normalization=True
    )
    #imgs = []
    #for i in test_df["Filepath"]:
    #    image = tf.keras.preprocessing.image.img_to_array(ut.load_img(i, target_size=(244, 244)))
    #    imgs.append(image)
    #imgs = np.stack(imgs)
    #test_generator.fit(imgs)
    return test_generator
