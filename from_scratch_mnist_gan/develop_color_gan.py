from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D, Conv2D, MaxPooling2D
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
import sys, os
import cv2
import argparse
os.environ["KERAS_BACKEND"] = "tensorflow"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12000,
                        help='the number of training steps to take')
    parser.add_argument('--batch', type=int, default=20,
                        help='the batch size')
    # parser.add_argument('--display', type=int, default=0,
    #                     help='display live with opencv')
    # parser.add_argument ('--data', type=str, default='flower',
    #                     help='data to parse, ****_sprites should be input, ***_output should be output)')
    parser.add_argument('--input', type=str, default='flower_sprites',
                        help='directory of examples (within colors)')
    parser.add_argument('--output', type=str, default='flower_generated',
                        help='directory of output (within colors)')
    parser.add_argument('--plot-loss-every', type=int, default=20,
                            help='how many epochs between saving the graph')
    parser.add_argument('--save-image-every', type=int, default=5,
                                help='how many epochs between printing image')
    parser.add_argument('--channels', type=int, default=3,
                        help='color:3 bw: 1')
    parser.add_argument('--display', dest='display', action='store_true')
    parser.add_argument('--no-display', dest='display', action='store_false')
    parser.set_defaults(display=True)
    # parser.add_argument('--display', type =bool, default=False,
    #                     help='display live with opencv')
    return parser.parse_args()

args = parse_args()

# input_dir = "color/" + args.data + "_sprites"
# output_dir = "color/" + args.data + "_output"

input_dir = "color/" + args.input
output_dir = "color/" + args.output


#change this directory to where hdf5 file is stored
DATASETS_DIR = os.path.dirname(os.path.realpath(__file__))

def loadMNIST(dataType):
    #parameter determines whether data is training or testing
    size = 10000
    f = h5py.File(DATASETS_DIR + "/data/mnist.hdf5", 'r')
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
    generatedImages = arr.reshape(len(arr), imageDim, imageDim)

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
    plt.savefig(output_dir + '/from_MNIST_dataset%d.png' %j)
    plt.close()


def generateImage(arr):
    magnification = 10
    img = arr
    res = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
    return res

def visualizeOne():
    arr = generator.predict(seed)
    res = generateImage(arr[0])
    cv2.imshow('Generated Image', res) # on windows, i had to install xming
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)

def visualizeTest(arr):
    res = generateImage(arr)
    cv2.imshow('Generated Image', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)

def getImageDim():
    if(input_dir == "data/mnist.hdf5"):
      return 28
    try:
        files = os.listdir(input_dir)
    except:
        print ("cannot load directory: " + input_dir)
        sys.exit(0)
    height, width, channels = cv2.imread(os.path.join(input_dir,files[0])).shape
    print ("height: ", height, " width: ", width, " channels: ", channels)
    #returns height of first image
    return height


def loadPixels():
    if(input_dir == "data/mnist.hdf5"):
      return loadMNIST("train")
    files = os.listdir(input_dir)
    count = len(files)
    images = np.empty((count, imageDim, imageDim, channels))
    for i in range(count):
        pic = cv2.imread(os.path.join(input_dir,files[i]))

        images[i] = pic
        if args.display:
            visualizeTest(pic)
    return images/255
    #return images
    #return np.reshape(images ,(count, imageDim**2 * 3))/255


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


imageDim = getImageDim()
channels = args.channels
#defining noise vector size
noise_vect_size = 10
image_shape = (imageDim, imageDim, channels)

np.random.seed(1000)

# Optimizer
adam = Adam(lr=0.0001, beta_1=0.5)

#seed= np.random.rand(noise_vect_size)
seed = np.random.normal(0, 1, size=[1, noise_vect_size])

#testing sequential model
generator = Sequential()

image_shape = (imageDim, imageDim, channels)

#stacking layers on model
#generator.add(Conv2D(filters, kernel_size, strides=1,
generator.add(Dense(128, activation = 'sigmoid', input_dim=noise_vect_size, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(Dropout(.1))
generator.add(Dense(imageDim**2*channels, activation = 'sigmoid'))
generator.add(Dropout(.1))
generator.add(Reshape((imageDim, imageDim, channels), input_shape=(imageDim**2*channels,)))
generator.add(Conv2D(35, (3, 3), padding='same', activation = 'sigmoid'))
generator.add(Conv2D(channels, (3, 3), padding='same', activation = 'sigmoid'))
#generator.add(Flatten())

#compiling loss function and optimizer
generator.compile(loss = 'mse', optimizer = adam)

#create discriminator
discriminator = Sequential()
#discriminator.add(Reshape((imageDim, imageDim, 3), input_shape=(imageDim**2*3,)))
discriminator.add(Conv2D(128, (3, 3), padding='same', input_shape=(image_shape)))
discriminator.add(MaxPooling2D(pool_size=(2, 2)))
discriminator.add(Conv2D(256, (3, 3), padding='same'))
discriminator.add(MaxPooling2D(pool_size=(2, 2)))
discriminator.add(Flatten())

discriminator.add(Dense(64, activation = 'sigmoid', input_dim=imageDim**2*channels, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
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
    plt.savefig(output_dir + '/gan_loss_epoch_%d.png' % epoch)
    plt.close()
    print ("Saving loss graph as "+ output_dir + "/gan_loss_epoch_%d.png" % epoch)


#method for creating batches of trainable data and training
def trainGAN(train_data, epochs=20, batch_size=10000):
    batchCount = len(train_data) / batch_size
    #loop for number of epochs
    # new_learning_rate = 0.0002

    for e in range(1, epochs+1):
        #loop for total number of batches
        print ('Epoch:', e)
        print ('Batches per epoch:', batchCount)
        for b in range(len(train_data)//batch_size):
            chosen_data_indexes = np.random.randint(1,train_data.shape[0],size = batch_size)
            data_x = np.array([train_data[i] for i in chosen_data_indexes]) #get next batch of the right size form training data and converts it to np.array
            data_x = np.reshape(data_x, (batch_size, imageDim, imageDim, channels))

            #train discriminator
            generated_x = generator.predict(np.random.random((batch_size, noise_vect_size)))#could use np.random.normal if training fails
            discriminator_x = np.concatenate((data_x, generated_x))#concatenate takes a tuple as input
            generated_x = generator.predict(np.random.random((batch_size, noise_vect_size)))#could use np.random.normal if training fails
            discriminator_x = np.concatenate((data_x, generated_x)) #concatenate takes a tuple as input
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
            if args.display:
                visualizeOne()

        # if e % 20 == 0 and e != 0:
        #     new_learning_rate -= 0.00001
        #     print("NEW LEARNING RATE IS: ", new_learning_rate)
        #     adam = Adam(lr=new_learning_rate, beta_1=0.5)
        #     gan.compile(loss = 'binary_crossentropy', optimizer = 'adam')

        if gloss < dloss:
          saveGeneratedImage(e, True)
        # if e % 10 == 0:
        #   saveGeneratedImage(e)


        dLosses.append(dloss)
        gLosses.append(gloss)
        print("Discriminator loss: ", dloss)
        print("Generator loss: ", gloss)
        if e % args.save_image_every == 0:
             #saveAlbum(e)
             saveGeneratedImage(e)
        if e % args.plot_loss_every == 0:
            plotLoss(e)

    plotLoss(e)

    return


magnification = 10

#generate mnist input images_original
def plotMNISTInput(arr, dim=(10, 10), figsize=(10, 10), numberOfFpngs=100):
    #look at input MNIST
    print("should be generating image")
    generatedImages = arr.reshape(len(arr), imageDim, imageDim)

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
    plt.savefig(output_dir + '/from_MNIST_dataset%d.png' %j)
    plt.close()


def generateImage(arr):
    magnification = 10
    #img = np.reshape(arr, (imageDim, imageDim, 3))
    img = arr
    res = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
    return res

def visualizeOne():
    arr = generator.predict(seed)
    res = generateImage(arr[0])
    cv2.imshow('Generated Image', res) # on windows, i had to install xming
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)

def visualizeTest(arr):
    #print ("arr is: ", arr)
    #print ("image shape is: ", arr.shape)
    res = generateImage(arr)
    #print ("new shape is: ", res.shape)
    #cv2.imshow('Generated Image', res)
    cv2.imshow('Generated Image', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)

# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
        noise = np.random.normal(0, 1, size=[examples, noise_vect_size])
        generatedImages = generator.predict(noise)
        generatedImages = generatedImages.reshape(examples, imageDim, imageDim, channels)

        plt.figure(figsize=figsize) #figsize is number of images
        for i in range(generatedImages.shape[0]):
                plt.subplot(dim[0], dim[1], i+1)
                plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
                plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_dir + '/gan_generated_image_epoch_%d.png' % epoch)
        plt.close()

def saveGeneratedImage(e, low_loss=False):
  arr = generator.predict(seed)
  # img = np.reshape(arr, (image_shape))
  # img = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
  # img = img*255
  # img = img.astype(np.uint8)
  img = generateImage(arr[0])*255
  if low_loss:
    cv2.imwrite(output_dir + '/low_loss_generated_image_epoch_%d.png' % e, img)
  else:
    cv2.imwrite(output_dir + '/generated_image_epoch_%d.png' % e, img)

#save a bunch of random images
def saveAlbum(e, shape = (3,3)):
    #noise = np.random.normal(0, 1, size=[shape+noise_shape])
    collage = np.empty (shape = (shape[0]*image_shape[0],shape[1]*image_shape[1],image_shape[2]))
    print ("combined image shape is: ", collage.shape)
    for x in range (shape[0]):
        for y in range (shape[1]):
            noise = np.random.random(shape+(1,)+noise_shape)
            image = generator.predict(noise[x,y])
            print ("image shape is: ", image.shape )
            #place pixel values of image in the collage
            collage[x*image_shape[0]:(x+1)*image_shape[0],y*image_shape[1]:(y+1)*image_shape[1]] = (image)
        # for y in range (shape[1])
        #     image = generator.predict(noise[x,y])
        #     img = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
        #     img = img*255
    collage *= 255
    cv2.imwrite(output_dir + '/many_%d_epoch_%d.png' % (shape[0]*shape[1], e), collage)

def printIntro():
    print("input from: ", input_dir, " output from: ", output_dir)
    print("batch size: ", batch_size, " epochs: ", epochs)


#grabbing all training inputs and begin training
if __name__ == '__main__':
    epochs = args.epochs
    batch_size = args.batch
    x_train = loadPixels()
    printIntro()
    trainGAN(x_train, epochs = epochs, batch_size=batch_size)
