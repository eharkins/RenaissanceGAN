from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
import keras.utils
import numpy as np
import h5py
import math
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import cv2
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from scipy.misc import imsave


#side of each image
imageDim = 28

#change this directory to where hdf5 file is stored
DATASETS_DIR = os.path.dirname(os.path.realpath(__file__))


def loadFaces():
    size = 2429
    #size = 10000
    f = h5py.File("faces.hdf5", 'r')
    X = f['data'][:size]
    return X


def loadMNIST(dataType):
	#parameter determines whether data is training or testing
	size = 10000
	f = h5py.File(DATASETS_DIR + "/mnist.hdf5", 'r')
	X = f['x_'+dataType][:size]
	maxes = X.max(axis=0)
	for i in range(len(maxes)):
		if maxes[i] == 0:
			maxes[i] = 0.1
	X *= 1/maxes
		# print X.shape

	print ("MNIST Dataset LOADED")

	return X

#generate mnist input images_original
def plotMNISTInput(arr, dim=(10, 10), figsize=(10, 10), numberOfFpngs=100):
	#look at input MNIST
	print("should be generating image")
	generatedImages = arr.reshape(len(arr), 28, 28)

	plt.figure(figsize=figsize)
	for j in range(100):
		plt.figure(figsize=figsize)
		i=0
		print(i)
		for i in range(generatedImages.shape[0]//numberOfFpngs):
			plt.subplot(dim[0], dim[1], i+1)
			plt.imshow(generatedImages[i+j*numberOfFpngs], interpolation='nearest', cmap='gray_r')
			plt.axis('off')
		plt.tight_layout()
		plt.savefig('images/from_MNIST_dataset%d.png' %j)


#defining noise vector size
noise_vect_size = 784

np.random.seed(1000)

# Optimizer
adam = Adam(lr=0.00004, beta_1=0.5)

#testing sequential model
generator = Sequential()

#stacking layers on model
generator.add(Dense(35, activation = 'sigmoid', input_dim=noise_vect_size, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
# generator.add(Dense(35, activation = 'sigmoid'))
generator.add(Dense(imageDim**2, activation = 'sigmoid'))

#compiling loss function and optimizer
generator.compile(loss = 'mse', optimizer = adam)

#create discriminator
discriminator = Sequential()

discriminator.add(Dense(35, activation = 'sigmoid', input_dim=imageDim**2, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(Dense(35, activation = 'sigmoid'))
discriminator.add(Dense(1, activation = 'sigmoid'))

#compiling loss function and optimizer
discriminator.compile(loss = 'mse', optimizer = adam)

#creating the combined model
discriminator.trainable = False
gan_input = Input(shape=(noise_vect_size,))
discrimInput = generator(gan_input)
gan_output = discriminator(discrimInput)
gan = Model(inputs = gan_input, outputs = gan_output)
gan.compile(loss = 'mse', optimizer = adam)

dLosses = []
gLosses = []

def plotLoss(epoch):
	plt.figure(figsize=(10, 8))
	plt.plot(dLosses, label='Discriminitive loss')
	plt.plot(gLosses, label='Generative loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig('images/gan_loss_epoch_%d.png' % epoch)
	print ("Saving loss graph as images/gan_loss_epoch_%d.png" % epoch)


#method for creating batches of trainable data and training
def trainGAN(train_data, epochs=20, batch_size=10000):

	batchCount = len(train_data) / batch_size
	#loop for number of epochs
	new_learning_rate = 0.00004
	oldGloss = 100
	increasing_epoch_counter = 0
	for e in range(epochs):
		#loop for total number of batches
		print ('Epoch:', e)
		print ('Batches per epoch:', batchCount)
		for b in range(len(train_data)//batch_size):
			chosen_data_indexes = np.random.randint(1,train_data.shape[0],size = batch_size)
			data_x = np.array([train_data[i] for i in chosen_data_indexes]) #get next batch of the right size form training data and converts it to np.array

			#train discriminator
			generated_x = generator.predict(np.random.random((batch_size, noise_vect_size)))#could use np.random.normal if training fails
			# gan.compile(loss = 'binary_crossentropy', optimizer = 'adam')
			discriminator_x = np.concatenate((data_x, generated_x))#concatenate takes a tuple as input
			discriminator_y = np.zeros(2*batch_size)
			discriminator_y[:batch_size] = 0.9
			discriminator.trainable = True
			dloss = discriminator.train_on_batch(discriminator_x,discriminator_y)

			#train generator
			discriminator.trainable=False
			# gan.compile(loss = 'binary_crossentropy', optimizer = 'adam')
			gan_x = np.random.random((batch_size,noise_vect_size))
			gan_y = np.ones(batch_size) #creates an array of ones (expected output)
			gloss = gan.train_on_batch(gan_x, gan_y)
			visualizeOne()

		if gloss < dloss:
			arr = generator.predict(seed)
			img = np.reshape(arr, (imageDim, imageDim))
			img = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
			img = img*255
			img = img.astype(np.uint8)
			imsave('images/low_loss_generations/generated_image_epoch_%d.png' % e, img)

		if gloss > oldGloss:
			increasing_epoch_counter += 1
			if increasing_epoch_counter == 3:
				new_learning_rate = new_learning_rate/2
				print("NEW LEARNING RATE IS: ", new_learning_rate)
				adam = Adam(lr=new_learning_rate, beta_1=0.5)
				gan.compile(loss = 'mse', optimizer = 'adam')
				increasing_epoch_counter = 0

		oldGloss = gloss		

		dLosses.append(dloss)
		gLosses.append(gloss)
		print("Discriminator loss: ", dloss)
		print("Generator loss: ", gloss)
			# if e == 1 or e % 5 == 0:
		#      plotGeneratedImages(e)
		#      saveModels(e)

	plotLoss(e)

	return


magnification = 10

#seed= np.random.rand(noise_vect_size)
seed = np.random.normal(0, 1, size=[1, noise_vect_size])
print ("seed: ", seed.shape)

# arr = generator.predict(seed)
# # print ("shape of arr: ",arr.shape)
# res = generateImage(arr)
# cv2.imwrite("images/low_loss_generations/generated_image_epoch_%d.png" % e, res)


def generateImage(arr):
	img = np.reshape(arr, (imageDim, imageDim))
	res = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
	return res

def visualizeOne():
	arr = generator.predict(seed)
	# print ("shape of arr: ",arr.shape)
	res = generateImage(arr)
	cv2.imshow('Generated Image', res)
	# print ("showing image")
	if cv2.waitKey(1) & 0xFF == ord('q'):
		sys.exit(0)
	# print ("drawn")
# time.sleep(.1)


# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
		noise = np.random.normal(0, 1, size=[examples, noise_vect_size])
		generatedImages = generator.predict(noise)
		generatedImages = generatedImages.reshape(examples, imageDim, imageDim)

		plt.figure(figsize=figsize)
		for i in range(generatedImages.shape[0]):
				plt.subplot(dim[0], dim[1], i+1)
				plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
				plt.axis('off')
		plt.tight_layout()
		plt.savefig('images/gan_generated_image_epoch_%d.png' % epoch)

#grabbing all training inputs and begin training
if __name__ == '__main__':
	epochs = int(sys.argv[1])
	batch_size = int(sys.argv[2])
	x_train = loadMNIST("train")
	# x_train = loadFaces()
	trainGAN(x_train, epochs = epochs, batch_size=batch_size)
