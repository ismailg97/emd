import tensorflow as tf

def resblock(x, filters, strides):
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

def build_general_model(input, nr_classes):
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
    if nr_classes > 1:
        out = tf.keras.layers.Dense(nr_classes, activation="relu")(out)
        return tf.keras.layers.Softmax()(out)
    elif nr_classes == 1:
        return tf.keras.layers.Dense(nr_classes, activation="relu")(out)
    else:
        raise ValueError('The Number of Classes is invalid.')


