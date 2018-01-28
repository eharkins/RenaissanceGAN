from keras.models import Sequential
from keras.layers import Dense, Activation
import keras.utils
import numpy as np
import h5py
import math
import random

#change this directory to where hdf5 file is stored
DATASETS_DIR = "C:/Users/neel/Documents/Thesis Stuffs/"
def loadMNIST(dataType):
	#parameter determines whether data is training or testing
	size = 10000
	f = h5py.File(DATASETS_DIR + "mnist.hdf5", 'r')
	X = f['x_'+dataType][:size]

	maxes = X.max(axis=0)
	for i in range(len(maxes)):
		if maxes[i] == 0:
			maxes[i] = 0.1
	X *= 1/maxes
    # print X.shape

	raw_y = np.array([f['t_'+dataType][:size]]).T

	y = []
	for row in raw_y:
		y.append(convertToOneHot(row[0], 10))
	
	y = np.array(y)

	print ("MNIST Dataset LOADED")
	
	return X, y

def convertToOneHot(val, size):
    x = np.zeros(size)
    x[val] = 0.9
    return x

#testing sequential model
generator = Sequential()

#stacking layers on model
generator.add(Dense(256, activation = 'relu', input_dim=128))
generator.add(Dense(512, activation = 'relu'))
generator.add(Dense(1024, activation = 'relu'))
generator.add(Dense(784, activation = 'relu'))
generator.add(Activation('softmax'))

#create discriminator
discriminator = Sequential()

discriminator.add(Dense(512, activation = 'relu', input_dim=784))
discriminator.add(Dense(256), activation = 'relu')
discriminator.add(Dense(128), activation = 'relu')
discriminator.add(Dense(64), activation = 'reul')
discriminator.add(Dense(8), activation = 'relu')
discriminator.add(Dense(1), activation = 'relu')
discriminator.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics =['accuracy'])