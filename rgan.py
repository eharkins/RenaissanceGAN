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

DATASETS_DIR = os.path.dirname(os.path.realpath(__file__))

# [1]
#https://github.com/Zackory/Keras-MNIST-GAN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12000, help='the number of training steps to take')
    parser.add_argument('--batch', type=int, default=5, help='the batch size')
    parser.add_argument('--input', type=str, default='bach.mid', help='directory of examples (within colors)')
    parser.add_argument('--output', type=str, default='uni_generated', help='directory of output (within colors)')
    parser.add_argument('--plot-every', type=int, default=100, help='how many epochs between saving the graph')
    parser.add_argument('--save-every', type=int, default=100, help='how many epochs between printing image')
    parser.add_argument('--display', dest='display', action='store_true', default=True)
    parser.add_argument('--no-display', dest='display', action='store_false')
    # parser.set_defaults(display=True)
    return parser.parse_args()

os.environ["KERAS_BACKEND"] = "tensorflow"

# Set args
args = parse_args()
data_source = "data/" + args.input
output_dir = "output/" + args.output
epochs = args.epochs
batch_size = args.batch
doing_music = 0

# define the image dimensions for training data
def getImageDim(data_source):
    try:
        files = os.listdir(data_source)
    except:
        print ("cannot load directory: " + data_source)
        sys.exit(0)
    height, width, channels = cv2.imread(os.path.join(data_source,files[0])).shape
    print ("height: ", height, " width: ", width, " channels: ", channels)
    return height

# process a dataset of images
def loadPixels(data_source):
    channels = 3 #RGB
    imageDim = getImageDim(data_source)
    data_shape = (imageDim, imageDim, 3)

    files = os.listdir(data_source)
    count = len(files)
    images = np.empty((count, imageDim, imageDim, channels))
    for i in range(count):
        pic = cv2.imread(os.path.join(data_source,files[i]))

        images[i] = pic
        if args.display and i < 100:
            visualize(pic)
    data = images/255
    return data, data_shape

# https://github.com/menwhitehead/NPANN/blob/a4e1c02efd454361191cc7384bad5f1866a7899d/npann/Utilities/misc_functions.py
# process the mnist dataset from h5py file
def loadMNIST(data_source, dataType, imageDim = 28):
    data_shape = (imageDim, imageDim, 1)
    size = 10000
    f = h5py.File(DATASETS_DIR +  "/" + data_source, 'r')
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
    if args.display:
        for i in range(50):
            visualize(X[i])
    print ("MNIST Dataset LOADED")
    return X, data_shape

# produce an image from output of generator
def generateImage(arr, magnification = 10):
    #deal with more or less than 3 channels
    img = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=arr.dtype)
    for i in range(arr.shape[2]):
        img[:,:,i%3] += arr[:,:,i]
    # create a musical score
    if doing_music:
        img = 1-img #flip color values for white background - helps musical score visibility
    res = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
    return res

# show an image on the screen
def visualize(arr):
    res = generateImage(arr)
    cv2.imshow('Renaissance GAN', res) # on windows, x server is needed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)

# [1]
# plot the loss from the generator and discriminator, along with discriminator's accuracy
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

# save the generated image
def saveImage(arr, e, low_loss=False):
  img = generateImage(arr[0])*255
  if low_loss:
    cv2.imwrite(output_dir + '/low_loss_generated_image_epoch_%d.png' % e, img)
  else:
    cv2.imwrite(output_dir + '/generated_image_epoch_%d.png' % e, img)

#save several images from seeds - this helps us see a variety of what the GAN is producing
def saveAlbum(generator, e, data_shape, seeds):
    shape = seeds.shape
    collage = np.empty (shape = (shape[0]*data_shape[0],shape[1]*data_shape[1],data_shape[2]))
    for x in range (shape[0]):
        for y in range (shape[1]):
            seed = seeds[x,y]
            arr = generator.predict(seed.reshape((1,100)))[0]
            image = generateImage(arr, 1)
            #place pixel values of image in the collage
            collage[x*data_shape[0]:(x+1)*data_shape[0],y*data_shape[1]:(y+1)*data_shape[1]] = (image)
    collage *= 255
    cv2.imwrite(output_dir + '/epoch_%d.png' % (e), collage)

# create a directory for the saved images, midi files, and graphs
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

noise_vect_size = 100

# uncomment for development: consistent seed
# np.random.seed(1000)

#seed for display
seed = np.random.normal(0, 1, size=[1, noise_vect_size])
#seed for multi-image generation
seeds = np.random.uniform(-1, 1, (3,3,noise_vect_size))

# save and show the encoded input songs as images
def showInputSongs(songs):
    debug_dir = output_dir + "/midi_input"
    writeCutSongs(songs, debug_dir)
    if args.display:
        initPygame();
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    for i in range(len(songs)):
        minisong = songs[i]
        res = generateImage((minisong))
        cv2.imwrite(output_dir+"/midi_input/input_score_%d.png" % i, res*255)
        if args.display:
            cv2.imshow('Renaissance GAN', res)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(0)

# if reading a midi file, process midi data
if data_source[-4:] == ".mid":
    print (" MUSIC! ")
    doing_music = 1
    # import the music processing methods
    from music_util import *
    # import the music model
    from models.midi_model import *
    # produce minisongs from the training data
    songs, shape = loadMidi(data_source)
    showInputSongs(songs)
    x_train, data_shape = songs, shape
elif data_source[-5:] == ".hdf5":
    print (" MNIST! ")
    x_train, data_shape = loadMNIST(data_source, "train")
elif os.listdir(data_source)[0][-4:] == ".mid":
    print (" MUSIC! (MULTIPLE SONGS)")
    doing_music = 1
    from music_util import *
    from models.midi_model import *
    songs, shape = loadManyMidi(data_source)
    showInputSongs(songs)
    x_train, data_shape = songs, shape
else:
    print (" COLOR IMAGES! ")
    x_train, data_shape = loadPixels(data_source)
if not doing_music:
    from models.upsample_model import *


channels = data_shape[2]
data_size = data_shape[0]*data_shape[1]*data_shape[2]

gan, generator, discriminator = makeGAN(data_shape, noise_vect_size)

# save a summary of the model configuration
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


# https://github.com/vwrs/dcgan-mnist
def trainGAN(train_data, epochs, batch_size):
    batchCount = len(train_data) / batch_size

    y_g = [1]*batch_size
    y_d_true = [1]*batch_size
    y_d_gen = [0]*batch_size

    for e in range(1, epochs+1):
        #loop for total number of batches
        print ('Epoch:', e)
        print ('Batches per epoch:', batchCount)
        for b in range(len(train_data)//batch_size):

            X_d_true = train_data[b*batch_size:(b+1)*batch_size]
            X_g = np.array([np.random.normal(0,0.5,100) for _ in range(batch_size)])
            X_d_gen = generator.predict(X_g, verbose=0)
            y_d_true = np.reshape(np.array(y_d_true),(batch_size,1))
            y_d_gen = np.reshape(np.array(y_d_gen),(batch_size,1))
            y_g = np.reshape(np.array(y_g),(batch_size,1))
            # train discriminator
            discriminator.train_on_batch(X_d_true, y_d_true)
            dloss, accuracy = discriminator.train_on_batch(X_d_gen, y_d_gen)

            # train generator
            gloss = gan.train_on_batch(X_g, y_g)

            if args.display:
                arr = generator.predict(seed)[0]
                visualize(arr)

            dLosses.append(dloss)
            gLosses.append(gloss)
            accuracies.append(accuracy)
        print("Discriminator loss: ", dloss)
        print("Generator loss: ", gloss)
        print("Accuracy: ", accuracy)

        if e % args.save_every == 0:
            arr = generator.predict(seed)
            if doing_music:
                saveMidi(arr, e, output_dir, args.display)
                saveImage(arr, e)
            else:
                saveAlbum(generator, e, data_shape, seeds)

        if e % args.plot_every == 0:
            arr = generator.predict(seed)
            plotLoss(e)

    plotLoss(e)
    return


printIntro()
saveSummary()
trainGAN(x_train, epochs = epochs, batch_size=batch_size) #begin training
