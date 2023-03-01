import tensorflow as tf


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

    for i, example in enumerate(train_set.take(10)):
        print('Image {} shape: {} label: {}'.format(i + 1, example[0].shape, example[1]))

    print('Total Number of Classes: {}'.format(num_classes))
    print('Total Number of Training Images: {}'.format(len(train_set)))
    print('Total Number of Validation Images: {} \n'.format(len(val_set)))

    return train_set, val_set, len(train_set), len(val_set)

