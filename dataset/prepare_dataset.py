import os
import tensorflow as tf



"""Load and prepare the given dataset"""


def load_dataset(dataset_path, img_size, batch_size, val_split_size):
    # (train_set, val_set), dataset_info = tfds.load(
    #   dataset_name,
    #   split=['train[:80%]', 'train[80%:]'],
    #   with_info=True,
    #   as_supervised=True)

    train_set = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=val_split_size,
        subset="training",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size)

    val_set = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=val_split_size,
        subset="validation",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size)

    num_classes = 5  # dataset_info.features['label'].num_classes

    num_training_examples = 0
    num_validation_examples = 0

    for example in train_set:
        num_training_examples += 1

    for example in val_set:
        num_validation_examples += 1

    for i, example in enumerate(train_set.take(10)):
        print('Image {} shape: {} label: {}'.format(i + 1, example[0].shape, example[1]))

    print('Total Number of Classes: {}'.format(num_classes))
    print('Total Number of Training Images: {}'.format(num_training_examples))
    print('Total Number of Validation Images: {} \n'.format(num_validation_examples))

    return train_set, val_set, num_training_examples, num_validation_examples


# load_dataset()
