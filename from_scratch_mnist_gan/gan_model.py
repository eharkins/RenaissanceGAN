# describes the shape of the GAN

import numpy as np
import math, random

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import initializers, metrics

# Optimizer
adam = Adam(lr=0.0001, beta_1=0.5)


def makeGenerator(data_shape, noise_vect_size):
    data_size = data_shape[0]*data_shape[1]*data_shape[2]
    channels = data_shape[2]
    generator = Sequential()
    generator.add(Dense(128, activation = 'sigmoid', input_dim=noise_vect_size, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    #could random normal be responsible for lack of color variation?
    #generator.add(Dropout(.1))
    #generator.add(Dense(data_size, activation = 'sigmoid'))
    #generator.add(Dense(data_size, activation = 'sigmoid', input_dim=noise_vect_size, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    #generator.add(Dropout(.1))
    generator.add(Dense(data_size, activation = 'sigmoid'))
    #generator.add(Dropout(.1))
    generator.add(Reshape((data_shape), input_shape=(data_size,)))
    #generator.add(Conv2DTranspose(64, (3, 3), strides = (1,1), padding='same', activation = 'sigmoid'))
    #generator.add(Conv2DTranspose(channels, (3, 3), strides = (1,1), padding='same', activation = 'sigmoid'))
    generator.compile(loss = 'binary_crossentropy', optimizer = adam)
    return generator

def makeDiscriminator(data_shape):
    #create discriminator
    data_size = data_shape[0]*data_shape[1]*data_shape[2]
    discriminator = Sequential()
    discriminator.add(Conv2D(64, (3, 3), padding='none', input_shape=(data_shape), activation = 'sigmoid'))
    discriminator.add(MaxPooling2D(pool_size=(2, 2))) #max pooling is very important! without it, the GAN takes longer and produces only noise
    discriminator.add(Conv2D(64, (3, 3), padding='none', input_shape=(data_shape), activation = 'sigmoid'))
    discriminator.add(MaxPooling2D(pool_size=(2, 2)))
    discriminator.add(Conv2D(64, (3, 3), padding='none', input_shape=(data_shape), activation = 'sigmoid'))
    discriminator.add(MaxPooling2D(pool_size=(2, 2)))
    discriminator.add(Flatten())
    discriminator.add(Dense(32, activation = 'sigmoid', input_dim=data_size, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(Dense(1, activation = 'sigmoid'))
    discriminator.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    discriminator.trainable = False
    return discriminator

def makeGAN(data_shape, noise_vect_size = 10):
    generator = makeGenerator(data_shape, noise_vect_size)
    discriminator = makeDiscriminator(data_shape)
    gan_input = Input(shape=(noise_vect_size,))
    discrimInput = generator(gan_input)
    gan_output = discriminator(discrimInput)
    gan = Model(inputs = gan_input, outputs = gan_output)
    gan.compile(loss = 'binary_crossentropy', optimizer = adam)
    return gan, generator, discriminator
