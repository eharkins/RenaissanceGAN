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
model = Sequential()

#stacking layers on model
model.add(Dense(32, activation = 'relu', input_dim=784))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Activation('softmax'))

#configure learning process using .compile()
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

#generating dummy data
#data is 784*1000 and should represent brightness, normalized by softmax activation I think?
#labels is 1000*1 and from 0 to 9 representing what number is drawn

#data = np.random.random((1000,784))
#labels = np.random.randint(10, size=(1000,1))

#covert labels into one hot labels
#one_hot_labels = keras.utils.to_categorical(labels, num_classes=10)

#utilize matthew's hdf5 reading method that converts to one hot
data, one_hot_labels = loadMNIST('train')
testData, testLabels = loadMNIST('test')


#x_train and y_train are Numpy arrays
model.fit(data, one_hot_labels, epochs=10, batch_size=10)

#inputting batches for training directly 
#model.train_on_batch(x_batch, y_batch)

print("finished training randomly")
score = model.evaluate(testData, testLabels, batch_size=5)

print("loss: %s, accuracy: %d"% score)