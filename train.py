import os
import tensorflow as tf
import datetime
import keras
import matplotlib.pyplot as plt
from dataset.prepare_dataset import load_dataset
from models.generator import generator_model
from models.discriminator import discriminator_model


"""Define required parameters and hyper-parameters:"""

batch_size = 8
initial_lr = 0.0001
lr_decay_factor = 0.1
patience_factor = 3
number_of_epochs = 200
save_path = 'output/flowers/'
dataset_path = 'dataset/flower_photos'
img_size = 64
dataset_name = 'tf_flowers'
val_split_size = 0.2

hyper_params_dictionary = {'initial_lr': initial_lr,
                           'batch_size': batch_size,
                           'img_size': img_size,
                           'number_of_epochs': number_of_epochs,
                           'dataset_name': dataset_name,
                           'img_size': img_size}

with open(os.path.join(save_path, "hyperParams.txt"), "w+") as file:
    file.write(str(hyper_params_dictionary))

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss

    return total_loss


def generator_loss(fake_output):
    fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return fake_loss


def save_checkpoint(save_path, generator, discriminator, generator_optimizer, discriminator_optimizer):
    checkpoint_dir = save_path
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    return checkpoint_dir, checkpoint, checkpoint_prefix


def generate_and_save_images(model, epoch, seed, save_path):
    predictions = model(seed, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(save_path + '/' + f'{epoch:04d}.png')
    # plt.show()
    plt.close(fig)


def prepare_image(image, label):
    image = tf.image.resize(image, (img_size, img_size)) / 255.0
    return image, label


current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = save_path + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

train_set, val_set, train_num_example, val_num_example = load_dataset(dataset_path, img_size,
                                                                      batch_size, val_split_size)

train_batches = train_set.shuffle(train_num_example // 4).map(prepare_image).batch(batch_size).prefetch(1)
validation_batches = val_set.map(prepare_image).batch(batch_size).prefetch(1)

if not os.path.exists(save_path):
    os.makedirs(save_path)

noise = tf.random.normal([1, 100])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=patience_factor,
    decay_rate=lr_decay_factor)

generator = generator_model(100, img_size)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

discriminator = discriminator_model(3, img_size)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

checkpoint_dir, checkpoint, checkpoint_prefix = save_checkpoint(save_path,
                                                                generator,
                                                                discriminator,
                                                                generator_optimizer,
                                                                discriminator_optimizer)


def train(dataset, epochs):
    for epoch in range(epochs):
        print('Epoch {} started'.format(epoch))

        for image_batch, label_batch in dataset:
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # print('Start to generate image.....')
                generated_images = generator(noise, training=True)

                real_output = discriminator(tf.transpose(image_batch[0], perm=[0, 3, 1, 2]), training=True)
                # real_output = discriminator(image_batch[0], training=True)
                generated_images = tf.transpose(generated_images, [0, 3, 1, 2])
                fake_output = discriminator(generated_images, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss,
                                                       generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                            discriminator.trainable_variables)
            generator_optimizer.apply_gradients(
                zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, discriminator.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('gen_loss', gen_loss, step=epoch)
            print('generator loss: {}'.format(gen_loss))
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)
            print('discriminator loss: {}'.format(disc_loss))

        generate_and_save_images(generator,
                                 epoch + 1,
                                 noise,
                                 save_path)

        if epoch % 2 == 0:
            print('Saving checkpoint.....')
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Epoch {epoch} finished.')
        print('\n')

    # final epoch
    generate_and_save_images(generator,
                             epochs,
                             noise,
                             save_path)


if __name__ == '__main__':
    train(train_batches, number_of_epochs)
