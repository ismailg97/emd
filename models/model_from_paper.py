import tensorflow as tf


class PaperModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same")
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same")
        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")
        self.conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same")
        self.conv6 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")
        self.conv7 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same")
        self.conv8 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same")
        self.add = tf.keras.layers.Add()
        self.maxPool = tf.keras.layers.MaxPool2D(strides=(2, 2))
        self.globalAvgPooling = tf.keras.layers.GlobalAveragePooling2D()
        self.normalization = tf.keras.layers.BatchNormalization(axis=3)
        self.ReLU = tf.keras.layers.Activation("relu")
        #self.softMax = tf.keras.activations.softmax()
        self.dense = tf.keras.layers.Dense(1, activation="relu")

    def call(self, inputs=tf.keras.Input(shape=(224, 224), batch_size=32), training=None, mask=None):
        out0 = self.conv0(inputs)
        #out0 = self.maxPool(out0)

        out = self.conv1(out0)
        out = self.normalization(out)
        out = self.ReLU(out)
        out = self.conv1(out)
        out = self.normalization(out)
        out1 = self.add([out, out0])
        out1 = self.ReLU(out1)

        out = self.conv2(out1)
        out = self.normalization(out)
        out = self.ReLU(out)
        out = self.conv2(out)
        out = self.normalization(out)
        out2 = self.add([out, out1])
        out2 = self.ReLU(out2)

        out = self.conv3(out2)
        out = self.normalization(out)
        out = self.ReLU(out)
        out = self.conv3(out)
        out = self.normalization(out)
        out3 = self.add([out, out2])
        out3 = self.ReLU(out3)

        out = self.conv4(out3)
        out = self.normalization(out)
        out = self.ReLU(out)
        out = self.conv4(out)
        out = self.normalization(out)
        out4 = self.add([out, out3])
        out4 = self.ReLU(out4)

        out = self.conv5(out4)
        out = self.normalization(out)
        out = self.ReLU(out)
        out = self.conv5(out)
        out = self.normalization(out)
        out5 = self.add([out, out4])
        out5 = self.ReLU(out5)

        out = self.conv6(out5)
        out = self.normalization(out)
        out = self.ReLU(out)
        out = self.conv6(out)
        out = self.normalization(out)
        out6 = self.add([out, out5])
        out6 = self.ReLU(out6)

        out = self.conv7(out6)
        out = self.normalization(out)
        out = self.ReLU(out)
        out = self.conv7(out)
        out = self.normalization(out)
        out7 = self.add([out, out6])
        out7 = self.ReLU(out7)

        out = self.conv8(out7)
        out = self.normalization(out)
        out = self.ReLU(out)
        out = self.conv8(out)
        out = self.normalization(out)
        out8 = self.add([out, out7])
        out8 = self.ReLU(out8)

        out = self.globalAvgPooling(out8)
        return self.dense(out)
        #return self.softMax(out)




