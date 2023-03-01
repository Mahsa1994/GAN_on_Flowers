import tensorflow as tf
from tensorflow.keras import layers


def discriminator_model(input_channel, input_size):
    model = tf.keras.models.Sequential()

    model.add(layers.InputLayer((input_channel, input_size, input_size)))
    model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(filters=2 * 64, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(filters=4 * 64, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(filters=8 * 64, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(layers.Conv2D(filters=1, kernel_size=4, strides=2, padding='same'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

