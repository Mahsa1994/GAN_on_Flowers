# GAN Flowers
Train a simple GAN on flowers dataset.

## Dataset 
The flowers dataset includes 3670 labeled images in 5 classes. Dataset is splitted in 80% for training and 20% for test. It is done by the following line in the [train.py](train.py):

```python
train_set, val_set, train_num_example, val_num_example = load_dataset(dataset_path, img_size,
                                                                      batch_size, val_split_size)
```


The images of the dataset come in different sizes. Here is a sample of images:

<img src="https://github.com/Mahsa1994/GAN_Flowers/blob/main/sample1.png" width="50%">


## Defining the model
To train a model that generates flowers from the input noise, you need to define 2 main models: 
1) Discriminator
2) Generator

These two basically define generative adversarial networks (GANs). The principle is for the generator to generate a sample data point and for the discriminator to classify this generated sample as original or fake. The goal for the generator is to generate such high quality and close-to-real samples that the discriminator won't be able to distinguish between original or generated samples. By iteratively going back and forth between these two models, the generator eventually learns to make high-quality images.
The overall structure of a GAN is shown below: 

<img src="https://github.com/Mahsa1994/GAN_Flowers/blob/main/structure.jpeg" width="50%">

From the files [discriminator.py](models/discriminator.py) and [generator.py](models/generator.py), following are the definitions of the models:
```python
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
```

In order to train these models, we should come up with proper hyperparameters as training GANs can be quite tricky and hard. For that, the following hyperparameters were used (defined in [train.py](train.py)):
```python
batch_size = 8
initial_lr = 0.0001
lr_decay_factor = 0.1
patience_factor = 3
number_of_epochs = 200
save_path = 'output/flowers/'
dataset_path = 'dataset/flower_photos'
tensorboard_logger_path = '.'
weight_decay = 0.0001
img_size = 64
dataset_name = 'tf_flowers'
val_split_size = 0.2
```

## Running the code
The code has been tested and run using `python3.8`. First, a virtualenv shoud be created and the required packages should be installed:
‍‍
```shell
virtualenv -p python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
```

Now that the packages are installed, we can start the training:
```shell
python train.py
```

Checkpoints of the generator and the discriminator models will be saved by the `save_checkpoint()`. Also, samples of generated images will be saved by the `generate_and_save_images()`.
