# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.optimizers import Adam
from keras.layers import Flatten, Dropout

LR = 0.0002  # initial learning rate
B1 = 0.5  # momentum term
opt = Adam(lr=LR,beta_1=B1)

def makeGenerator(input_dim=100,units=1024,activation='relu'):
    print("generator")
    model = Sequential()
    model.add(Dense(input_dim=input_dim, units=units))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    print("model summary:")
    print(model.summary())
    return model

def makeDiscriminator(input_shape=(28, 28, 1),nb_filter=64):
    model = Sequential()
    model.add(Conv2D(nb_filter, (5, 5), strides=(2, 2), padding='same',
                            input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Conv2D(2*nb_filter, (5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(4*nb_filter))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print(model.summary())
    return model


def makeGAN(data_shape, noise_vect_size = 100):
    generator = makeGenerator(data_shape, noise_vect_size)
    discriminator = makeDiscriminator(data_shape)
    discriminator .trainable = True
    discriminator.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=opt)
    discriminator.trainable = False

    gan_input = Input(shape=(noise_vect_size,))
    discrimInput = generator(gan_input)
    gan_output = discriminator(discrimInput)
    gan = Model(inputs = gan_input, outputs = gan_output)
    gan.compile(loss = 'binary_crossentropy', optimizer = adam)
    return gan, generator, discriminator
