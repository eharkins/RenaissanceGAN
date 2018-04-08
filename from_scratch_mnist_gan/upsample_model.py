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

# a very lightly modified version of the mode originally used in dc gan.


def makeGenerator(data_shape, noise_vect_size=100, activation = 'relu'):
#def generator(input_dim=100,units=1024,activation='relu'):
    print("generator")
    data_size = data_shape[0]*data_shape[1]*data_shape[2]
    channels = data_shape[2]
    model = Sequential()
    model.add(Dense(input_dim=noise_vect_size, units=1024))
    model.add(BatchNormalization())
    model.add(Activation(activation))

    fractionx = fractiony = 4
    new_shape = (int(data_shape[0]/fractionx), int(data_shape[1]/fractiony), data_shape[2]*fractionx*fractiony)
    print ("new_shape is: ", new_shape)

    #model.add(Dense(128*7*7))
    model.add(Dense(data_size))

    model.add(BatchNormalization())
    model.add(Activation(activation))
    #model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(Reshape(new_shape, input_shape=(data_size,)))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(channels, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    print("model summary:")
    print(model.summary())
    return model

def makeDiscriminator(data_shape, nb_filter=64):
    model = Sequential()
    model.add(Conv2D(nb_filter, (5, 5), strides=(2, 2), padding='same',
                            input_shape=data_shape))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Conv2D(2*nb_filter, (5, 5), strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    #model.add(Dense(4*nb_filter))
    data_size = data_shape[0]*data_shape[1]*data_shape[2]
    model.add(Dense(data_size))

    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print(model.summary())
    return model


def makeGAN(data_shape, noise_vect_size = 100):

    g = makeGenerator(data_shape, noise_vect_size)
    d = makeDiscriminator(data_shape)
    # print ("data shape is: ", data_shape)
    # print("Generator ",g )
    opt = Adam(lr=LR,beta_1=B1)
    d.trainable = True
    d.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=opt)
    d.trainable = False
    gan = Sequential([g, d])
    gan.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer=opt)

    return gan, g, d
