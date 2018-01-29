from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
import keras.utils
import numpy as np
import h5py
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



#change this directory to where hdf5 file is stored
DATASETS_DIR = ""
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
	

	
#defining noise vector size
noise_vect_size = 128
	
#testing sequential model
generator = Sequential()

#stacking layers on model
generator.add(Dense(256, activation = 'relu', input_dim=noise_vect_size))
generator.add(Dense(512, activation = 'relu'))
generator.add(Dense(1024, activation = 'relu'))
generator.add(Dense(784, activation = 'softmax'))

#compiling loss function and optimizer
generator.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics =['accuracy'])

#create discriminator
discriminator = Sequential()

discriminator.add(Dense(512, activation = 'relu', input_dim=784))
discriminator.add(Dense(256, activation = 'relu'))
discriminator.add(Dense(128, activation = 'relu'))
discriminator.add(Dense(64, activation = 'relu'))
discriminator.add(Dense(8, activation = 'relu'))
discriminator.add(Dense(1, activation = 'softmax'))

#compiling loss function and optimizer
discriminator.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics =['accuracy'])

#creating the combined model
gan_input = Input(shape=(noise_vect_size,))
discrimInput = generator(gan_input)
gan_output = discriminator(discrimInput)
gan = Model(inputs = gan_input, outputs = gan_output)
gan.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics =['accuracy'])

#method for creating batches of trainable data and training
def trainGAN(train_data, train_labels, batch_size=10000, epochs=20):
	for e in range(epochs): #loop for number of epochs
		print("epoch: ",e)
		for b in range(len(train_data)//batch_size): #loop for total number of batches
			print(range(len(train_data)//batch_size))
			print("batch: ",b)
			chosen_data_indexes = np.random.randint(1,train_data.shape[0],size = batch_size)
			data_x = np.array([train_data[i] for i in chosen_data_indexes]) #get next batch of the right size form training data and converts it to np.array
			
			#train discriminatorr
			generated_x = generator.predict(np.random.random((batch_size, noise_vect_size)))#could use np.random.normal if training fails
			discriminator.trainable = True
			gan.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics =['accuracy'])
			discriminator_x = np.concatenate((data_x, generated_x))#concatenate takes a tuple as input
			discriminator_y = np.zeros(2*batch_size)
			discriminator_y[:batch_size] = 0.9
			discriminator.train_on_batch(discriminator_x,discriminator_y)
			
			#train generator
			discriminator.trainable=False
			gan.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics =['accuracy'])
			gan_x = np.random.random((batch_size,noise_vect_size))
			gan_y = np.ones(batch_size) #creates an array of ones (expected output)
			gan.train_on_batch(gan_x, gan_y)
	return

def save_generated_images(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
	noise = np.random.normal(0, 1, size=[examples, noise_vect_size])
	generatedImages = generator.predict(noise)
	generatedImages = generatedImages.reshape(examples, 28, 28)

	plt.figure(figsize=figsize)
	for i in range(generatedImages.shape[0]):
	    plt.subplot(dim[0], dim[1], i+1)
	    plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
	    plt.axis('off')
	plt.tight_layout()
	plt.savefig('images/gan_generated_image_epoch_%d.png' % epoch)

#grabbing all training inputs and labels from hdf5
x_train, y_train = loadMNIST("train")

#call training function for GAN
trainGAN(x_train, y_train, batch_size=10000, epochs = 20)
save_generated_images(1)
