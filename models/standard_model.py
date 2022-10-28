import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu")
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")
        self.maxPool = tf.keras.layers.MaxPool2D()
        self.globalAvgPooling = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(64, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(1, activation="relu")

    def call(self, inputs=tf.keras.Input(shape=(120, 120, 3)), training=None, mask=None):
        out = self.conv1(inputs)
        out = self.maxPool(out)
        out = self.conv2(out)
        out = self.maxPool(out)
        out = self.globalAvgPooling(out)
        out = self.dense1(out)
        out = self.dense2(out)
        return self.dense3(out)



    #Defining Model
    #inputs = tf.keras.Input(shape=(120, 120, 3))
    #x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), activation="relu")(inputs)
    #x = tf.keras.layers.MaxPool2D()(x)
    #x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    #x = tf.keras.layers.MaxPool2D()(x)
    #print(inputs)
    #print(x)
    #x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #print(x)
    #x = tf.keras.layers.Dense(64, activation="relu")(x)
    #x = tf.keras.layers.Dense(64, activation="relu")(x)
    #outputs = tf.keras.layers.Dense(1, activation="relu")(x)
    #model = tf.keras.Model(inputs=inputs, outputs= outputs)
