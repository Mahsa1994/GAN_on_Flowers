import tensorflow as tf
from tensorflow.keras import layers


def generator_model(input_size, output_size):
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(4 * 4 * 512, input_dim=input_size))
    model.add(layers.Reshape(target_shape=(4, 4, 512)))
    model.add(layers.Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same', activation='tanh'))
    model.add(layers.Reshape(target_shape=(output_size, output_size, 3)))
    return model
