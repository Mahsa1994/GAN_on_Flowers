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


## How to train your model?
To train a model that generates flowers from the input noise, you need to define 2 main models: 
1) Discriminator
2) Generator

These two basically define generative adversarial networks (GANs). The principle is for the generator to generate a sample data point and for the discriminator to classify this generated sample as original or fake. The goal for the generator is to generate such high quality and close-to-real samples that the discriminator won't be able to distinguish between original or generated samples. By iteratively going back and forth between these two models, the generator eventually learns to make high-quality images.
The overall structure of a GAN is shown below: 

<img src="https://github.com/Mahsa1994/GAN_Flowers/blob/main/structure.jpeg" width="50%">





