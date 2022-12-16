import tensorflow as tf


class PaperModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same")
        self.add = tf.keras.layers.Add()
        self.maxPool = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=(2,2))
        self.globalAvgPooling = tf.keras.layers.GlobalAveragePooling2D()
        self.normalization = tf.keras.layers.BatchNormalization()
        self.ReLU = tf.keras.layers.Activation("relu")
        #self.softMax = tf.keras.activations.softmax()
        self.dense = tf.keras.layers.Dense(1, activation="relu")

    def resblock(self, x, filters, strides):

        if(strides == 1):
            x_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2))(x)
            x_conv = tf.keras.layers.BatchNormalization()(x_conv)
            #x_conv = x
            fx = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same', strides=(2,2))(x)
            #fx = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
        else:
            #x_conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1, 1))(x)
            #x_conv = tf.keras.layers.BatchNormalization()(x_conv)
            x_conv = x
            fx = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)

        fx = tf.keras.layers.BatchNormalization()(fx)
        fx = tf.keras.layers.Activation("relu")(fx)
        fx = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(fx)
        out = tf.keras.layers.Add()([x_conv, fx])
        out = tf.keras.layers.BatchNormalization()(out)
        out = tf.keras.layers.ReLU()(out)
        print(x_conv.shape)
        print(fx.shape)
        return out

    def call(self, inputs, training=None, mask=None):
        out0 = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same")(inputs)
        print(out0.shape)
        #out0 = self.maxPool(out0)

        out1 = self.resblock(out0, filters=64,strides=0)
        out2 = self.resblock(out1, filters=64, strides=0)
        out3 = self.resblock(out2, filters=128, strides=1)
        out4 = self.resblock(out3, filters=128, strides=0)
        out5 = self.resblock(out4, filters=256, strides=1)
        out6 = self.resblock(out5, filters=256, strides=0)
        out7 = self.resblock(out6, filters=512, strides=1)
        out8 = self.resblock(out7, filters=512, strides=0)
        out = self.globalAvgPooling(out8)
        return self.dense(out)
        #return self.softMax(out)







