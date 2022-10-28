import tensorflow as tf


# Defining the ImageDataGenerator and what Preprocessing should be done to the images
def getTrainGenerator():
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )


def getTestGenerator():
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
    )