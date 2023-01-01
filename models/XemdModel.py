from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Callable, Union, List

import keras
import tensorflow as tf
from keras.metrics import Metric
from keras import Model
from keras.activations import linear, softmax
from keras.metrics import categorical_accuracy
from keras.optimizers import SGD, Optimizer, Adam
# from keras.optimizers.schedules import ExponentialDecay
from tensorflow import TensorShape

from loss_functions.emd import EmdWeightHeadStart, GroundDistanceManager, self_guided_earth_mover_distance
from models import operations


class XEMDModel(tf.keras.Model):
    def __init__(self, nr_classes, ground_distance_manager, emd_weight_head_start):
        super().__init__()
        self.nr_classes = nr_classes
        self.ground_distance_manager = ground_distance_manager
        self.emd_weight_head_start = emd_weight_head_start
        self.second_to_last_layer = None

        self.conv0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same")

        self.conv1a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.conv1b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')

        self.conv2a = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.conv2b = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same')

        self.conv3a = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(2, 2))
        self.conv3b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', strides=(2, 2))
        self.conv3c = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')

        self.conv4a = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.conv4b = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same')

        self.conv5a = tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(2, 2))
        self.conv5b = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=(2, 2))
        self.conv5c = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')

        self.conv6a = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.conv6b = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same')

        self.conv7a = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), strides=(2, 2))
        self.conv7b = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', strides=(2, 2))
        self.conv7c = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')

        self.conv8a = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.conv8b = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same')

        self.add = tf.keras.layers.Add()
        self.maxPool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))
        self.globalAvgPooling = tf.keras.layers.GlobalAveragePooling2D()
        self.normalization = tf.keras.layers.BatchNormalization()
        self.ReLU = tf.keras.layers.Activation("relu")
        self.softMax = tf.keras.layers.Softmax()
        self.dense = tf.keras.layers.Dense(self.nr_classes, activation="relu")

    def resblock(self, x, filters, strides):
        if strides == 1:
            x_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2))(x)
            x_conv = tf.keras.layers.BatchNormalization()(x_conv)
            fx = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(2, 2))(x)
        else:
            x_conv = x
            fx = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
        fx = tf.keras.layers.BatchNormalization()(fx)
        fx = tf.keras.layers.Activation("relu")(fx)
        fx = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(fx)
        out = tf.keras.layers.Add()([x_conv, fx])
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out)
        return out

    def call(self, inputs=tf.keras.Input(shape=(3,224, 224)), training=None, mask=None):
        out0 = self.conv0(inputs)

        x_conv = out0
        fx = self.conv1a(out0)
        fx = self.normalization(fx)
        fx = self.ReLU(fx)
        fx = self.conv1b(fx)
        out = self.add([x_conv, fx])
        out = self.normalization(out)
        out1 = self.ReLU(out)

        x_conv = out1
        fx = self.conv2a(out1)
        fx = self.normalization(fx)
        fx = self.ReLU(fx)
        fx = self.conv2b(fx)
        out = self.add([x_conv, fx])
        out = self.normalization(out)
        out2 = self.ReLU(out)

        x_conv = self.conv3a(out2)
        x_conv = self.normalization(x_conv)
        fx = self.conv3b(out2)
        fx = self.normalization(fx)
        fx = self.ReLU(fx)
        fx = self.conv3c(fx)
        out = self.add([x_conv, fx])
        out = self.normalization(out)
        out3 = self.ReLU(out)

        x_conv = out3
        fx = self.conv4a(out3)
        fx = self.normalization(fx)
        fx = self.ReLU(fx)
        fx = self.conv4b(fx)
        out = self.add([x_conv, fx])
        out = self.normalization(out)
        out4 = self.ReLU(out)

        x_conv = self.conv5a(out4)
        x_conv = self.normalization(x_conv)
        fx = self.conv5b(out4)
        fx = self.normalization(fx)
        fx = self.ReLU(fx)
        fx = self.conv5c(fx)
        out = self.add([x_conv, fx])
        out = self.normalization(out)
        out5 = self.ReLU(out)

        x_conv = out5
        fx = self.conv6a(out5)
        fx = self.normalization(fx)
        fx = self.ReLU(fx)
        fx = self.conv6b(fx)
        out = self.add([x_conv, fx])
        out = self.normalization(out)
        out6 = self.ReLU()(out)

        x_conv = self.conv7a(out6)
        x_conv = self.normalization(x_conv)
        fx = self.conv7b(out6)
        fx = self.normalization(fx)
        fx = self.ReLU(fx)
        fx = self.conv7c(fx)
        out = self.add([x_conv, fx])
        out = self.normalization(out)
        out7 = self.ReLU(out)

        x_conv = out7
        fx = self.conv8a(out7)
        fx = self.normalization(fx)
        fx = self.ReLU(fx)
        fx = self.conv8b(fx)
        out = self.add([x_conv, fx])
        out = self.normalization(out)
        out8 = self.ReLU()(out)

        out = self.globalAvgPooling(out8)

        out = self.dense(out)
        self.second_to_last_layer = out
        return self.softMax(out)
