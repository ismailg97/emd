import tensorflow as tf
import keras.utils as ut
import numpy as np
import keras

def preprocess_TrainImages(train_df):
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        validation_split=0.2
    )
    imgs = []
    for i in train_df["Filepath"]:
        image = tf.keras.preprocessing.image.img_to_array(keras.utils.load_img(i, target_size=(244, 244)))
        imgs.append(image)
    imgs = np.stack(imgs)
    train_generator.fit(imgs)
    return train_generator




def preprocess_TestImages(test_df):
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True
    )
    imgs = []
    for i in test_df["Filepath"]:
        image = tf.keras.preprocessing.image.img_to_array(ut.load_img(i, target_size=(244, 244)))
        imgs.append(image)
    imgs = np.stack(imgs)
    test_generator.fit(imgs)
    return test_generator