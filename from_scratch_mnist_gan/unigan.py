
from keras.datasets import mnist
from keras import backend as K
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
from music21 import midi, stream, pitch, note, tempo, chord
os.environ["KERAS_BACKEND"] = "tensorflow"

from gan_model import *
from music_util import *

#change this directory to where hdf5 file is stored so nikhil can function as a cross-compatible special snowflake
DATASETS_DIR = os.path.dirname(os.path.realpath(__file__))





os.environ["KERAS_BACKEND"] = "tensorflow"

args = parse_args()

data_source = "data/" + args.input
output_dir = "output/" + args.output
epochs = args.epochs
batch_size = args.batch
doing_music = 0

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
    parser.add_argument('--input', type=str, default='bach.mid',
                        help='directory of examples (within colors)')
    parser.add_argument('--output', type=str, default='uni_generated',
                        help='directory of output (within colors)')
    parser.add_argument('--plot-every', type=int, default=25,
                            help='how many epochs between saving the graph')
    parser.add_argument('--save-every', type=int, default=5,
                                help='how many epochs between printing image')
    parser.add_argument('--channels', type=int, default=3,
                        help='color:3 bw: 1')
    parser.add_argument('--display', dest='display', action='store_true')
    parser.add_argument('--no-display', dest='display', action='store_false')
    parser.set_defaults(display=True)
    # parser.add_argument('--display', type =bool, default=False,
    #                     help='display live with opencv')
    return parser.parse_args()


def getImageDim(data_source):
    try:
        files = os.listdir(data_source)
    except:
        print ("cannot load directory: " + data_source)
        sys.exit(0)
    height, width, channels = cv2.imread(os.path.join(data_source,files[0])).shape
    print ("height: ", height, " width: ", width, " channels: ", channels)
    #returns height of first image
    return height


def loadPixels(data_source):
    channels = 3
    imageDim = getImageDim(data_source)
    data_shape = (imageDim, imageDim, 3)

    files = os.listdir(data_source)
    count = len(files)
    images = np.empty((count, imageDim, imageDim, channels))
    for i in range(count):
        pic = cv2.imread(os.path.join(data_source,files[i]))

        images[i] = pic
        if args.display:
            visualize(pic)
    data = images/255
    return data, data_shape
    #return images
    #return np.reshape(images ,(count, imageDim**2 * 3))/255

def loadMNIST(data_source, dataType, imageDim = 28):
    data_shape = (imageDim, imageDim, 1)
    #parameter determines whether data is training or testing
    size = 10000
    f = h5py.File(DATASETS_DIR +  "/" + data_source, 'r')
    #f = h5py.File(DATASETS_DIR + "/data/mnist.hdf5", 'r')
    X = f['x_'+dataType][:size]
    full_shape = (size,) + data_shape
    X = X.reshape(full_shape)
    maxes = X.max(axis=(1, 2)) #normalizing based on maximum pixel value
    print ("x: ", X.shape)
    print ("maxesx: ", maxes.shape)
    for i in range(len(maxes)):
        if maxes[i] == 0:
            maxes[i] = 0.1
            X[i]*=1/maxes[i]
    #X *= 1/maxes
    if args.display:
        for i in range(50):
            visualize(X[i])
        # print X.shape

    print ("MNIST Dataset LOADED")

    return X, data_shape



def generateImage(arr, magnification = 10):
    #take first three tracks for music
    if (arr.shape[2] > 3):
        arr = arr[:,:,:3]
    img = arr
    res = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
    return res

def visualize(arr):
    res = generateImage(arr)
    cv2.imshow('Generated Image', res) # on windows, i had to install xming
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)



def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir + '/gan_loss_epoch_%d.png' % epoch)
    plt.close()
    print ("Saving loss graph as "+ output_dir + "/gan_loss_epoch_%d.png" % epoch)


def saveImage(arr, e, low_loss=False):
  img = generateImage(arr[0])*255
  if low_loss:
    cv2.imwrite(output_dir + '/low_loss_generated_image_epoch_%d.png' % e, img)
  else:
    cv2.imwrite(output_dir + '/generated_image_epoch_%d.png' % e, img)

#save a bunch of random images
def saveAlbum(generator, e, shape = (3,3)):
    #noise = np.random.normal(0, 1, size=[shape+noise_shape])
    collage = np.empty (shape = (shape[0]*data_shape[0],shape[1]*data_shape[1],data_shape[2]))
    for x in range (shape[0]):
        for y in range (shape[1]):
            noise = np.random.random(shape+(1,)+noise_shape)
            image = generator.predict(noise[x,y])
            #place pixel values of image in the collage
            collage[x*data_shape[0]:(x+1)*data_shape[0],y*data_shape[1]:(y+1)*data_shape[1]] = (image)
        # for y in range (shape[1])
        #     image = generator.predict(noise[x,y])
        #     img = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
        #     img = img*255
    collage *= 255
    cv2.imwrite(output_dir + '/many_%d_epoch_%d.png' % (shape[0]*shape[1], e), collage)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#defining noise vector size
noise_vect_size = 10
np.random.seed(1000)
#seed= np.random.rand(noise_vect_size)
seed = np.random.normal(0, 1, size=[1, noise_vect_size])
#seed = np.random.rand(1, noise_vect_size)

def loadData():
    if(data_source[-5:] == ".hdf5"):
        print (" MNIST! ")
        return loadMNIST(data_source, "train")
    if data_source[-4:] == ".mid":
        print (" MUSIC! ")
        song, shape = loadMidi(data_source)
        writeCutSongs(song)
        global doing_music
        doing_music = 1
        return song, shape
    else:
        print (" COLOR IMAGES! ")
        return loadPixels(data_source)

x_train, data_shape = loadData() #grabbing all training inputs

minisong_size = data_shape[1]
channels = data_shape[2]
print ("channels is: ", channels)
data_size = data_shape[0]*data_shape[1]*data_shape[2]

gan, generator, discriminator = makeGAN(data_shape, 10)


def saveSummary(filename = "info.txt"):
    file = open(output_dir + "/" + filename,"w")
    with open(output_dir + "/" + filename,"w") as fh:
        fh.write("generator:\n")
        generator.summary(print_fn=lambda x: fh.write(x + '\n'))
        fh.write("discriminator:\n")
        discriminator.summary(print_fn=lambda x: fh.write(x + '\n'))
        fh.write("\ninput from: "+ data_source)
        fh.write("\nbatch size: "+ str(batch_size) + " epochs: " + str(epochs))

def printIntro():
    print("input from: ", data_source, " output to: ", output_dir)
    print("batch size: ", batch_size, " epochs: ", args.epochs)

dLosses = []
gLosses = []
accuracies=[]

def trainGAN(train_data, epochs, batch_size):
    batchCount = len(train_data) / batch_size
    #loop for number of epochs
    # new_learning_rate = 0.0002
    for e in range(1, epochs+1):
        #loop for total number of batches
        print ('Epoch:', e)
        print ('Batches per epoch:', batchCount)
        for b in range(len(train_data)//batch_size):
            chosen_data_indexes = np.random.randint(1,train_data.shape[0],size = batch_size)
            data_x = np.array([train_data[i] for i in chosen_data_indexes]) #get next batch of the right size from training data
            #data_x = np.reshape(data_x, ((batch_size) +data_shape))

            #train discriminator
            generated_x = generator.predict(np.random.random((batch_size, noise_vect_size)))#could use np.random.normal if training fails
            discriminator_x = np.concatenate((data_x, generated_x)) #concatenate takes a tuple as input
            discriminator_y = np.zeros(2*batch_size)
            discriminator_y[:batch_size] = 1
            discriminator.trainable = True
            dloss, accuracy = discriminator.train_on_batch(discriminator_x,discriminator_y)

            #train generator
            discriminator.trainable=False
            gan_x = np.random.random((batch_size,noise_vect_size))
            gan_y = np.ones(batch_size) #creates an array of ones (expected output)
            gloss = gan.train_on_batch(gan_x, gan_y)

            if args.display:
                arr = generator.predict(seed)[0]
                visualize(arr)

        dLosses.append(dloss)
        gLosses.append(gloss)
        accuracies.append(accuracy)
        print("Discriminator loss: ", dloss)
        print("Generator loss: ", gloss)
        #print("Accuracy: ", accuracy)

        if e % args.save_every == 0:
             # saveModels(e)
             arr = generator.predict(seed)
             if doing_music:
                 saveMidi(arr, e, output_dir)
             saveImage(arr, e)
        if e % args.plot_every == 0:
            arr = generator.predict(seed)
            plotLoss(e)

    plotLoss(e)

    return


printIntro()
saveSummary();
trainGAN(x_train, epochs = epochs, batch_size=batch_size) #begin training
