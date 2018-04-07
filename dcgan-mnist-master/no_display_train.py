#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import matplotlib
from keras.datasets import mnist
from PIL import Image
from adaptable_model import discriminator, generator
from keras.models import Sequential
from keras.optimizers import SGD, Adam
#from visualizer import *
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DATASETS_DIR = os.path.dirname(os.path.realpath(__file__));
os.chdir(DATASETS_DIR)

BATCH_SIZE = 32
NUM_EPOCH = 500
LR = 0.0002  # initial learning rate
B1 = 0.5  # momentum term
output_dir = GENERATED_IMAGE_PATH = 'images/'
GENERATED_MODEL_PATH = 'models/'
#data = "faces"
data = "../from_scratch_mnist_gan/data/eyes32"

magnification = 10
def generateImage(arr, shape):
    img = np.reshape(arr, shape)
    res = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
    return res

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

def visualize(arr, shape):
    res = generateImage(arr, shape)
    cv2.imshow('Generated Image', res) # on windows, i had to install xming
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)

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
        #if args.display:
        #visualize(pic, data_shape)
    data = images/255
    return data, data_shape
    #return images
    #return np.reshape(images ,(count, imageDim**2 * 3))/255


def saveAlbum(generator, e, data_shape, seeds):
    #noise = np.random.normal(0, 1, size=[shape+noise_shape])
    shape = seeds.shape
    #print ("shape is: ", shape)
    collage = np.empty (shape = (shape[0]*data_shape[0],shape[1]*data_shape[1],data_shape[2]))
    for x in range (shape[0]):
        for y in range (shape[1]):
            #noise = np.random.random(shape+(1,)+noise_shape)
            #image = generator.predict(noise[x,y])
            seed = seeds[x,y]
            image = generator.predict(seed.reshape((1,100)))
            #place pixel values of image in the collage
            collage[x*data_shape[0]:(x+1)*data_shape[0],y*data_shape[1]:(y+1)*data_shape[1]] = (image)
        # for y in range (shape[1])
        #     image = generator.predict(noise[x,y])
        #     img = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
        #     img = img*255
    collage *= 255
    cv2.imwrite(output_dir + '/many_%d_epoch_%d.png' % (shape[0]*shape[1], e), collage)


def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    # plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir + '/gan_loss_epoch_%d.png' % epoch)
    plt.close()
    print ("Saving loss graph as "+ output_dir + "/gan_loss_epoch_%d.png" % epoch)



dLosses = []
gLosses = []
accuracies=[]

def train():
    #(X_train, y_train), (_, _) = mnist.load_data()

    X_train, data_shape = loadPixels(data)
    channels = data_shape[2]

    # normalize images- this is the line to look for


    #X_train = (X_train.astype(np.float32) - 127.5)/127.5
    #X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], channels)


    print("XTRAAAAIN")
    #print(X_train)
    #data_shape = (28, 28, 1)
    g = generator(data_shape)
    d = discriminator(data_shape)
    print ("data shape is: ", data_shape)
    print("Generator ",g )
    opt = Adam(lr=LR,beta_1=B1)
    d.trainable = True
    d.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=opt)
    d.trainable = False
    dcgan = Sequential([g, d])
    opt= Adam(lr=LR,beta_1=B1)
    dcgan.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer=opt)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    print("NUM BATCHES: ", num_batches)
    # create directory
    if not os.path.exists(GENERATED_IMAGE_PATH):
        os.mkdir(GENERATED_IMAGE_PATH)
    if not os.path.exists(GENERATED_MODEL_PATH):
        os.mkdir(GENERATED_MODEL_PATH)

    print("-------------------")
    print("Total epoch:", NUM_EPOCH, "Number of batches:", num_batches)
    print("-------------------")
    #z_pred = np.array([np.random.uniform(-1,1,100) for _ in range(25)])
    #z_pred.reshape((5, 5, 100))
    z_pred = np.random.uniform(-1,1,(5,5,100))
    print ("zzpred shape is: ", z_pred.shape)
    y_g = [1]*BATCH_SIZE
    y_d_true = [1]*BATCH_SIZE
    y_d_gen = [0]*BATCH_SIZE
    seed = np.random.normal(0, 1, size=[1, 100])
    for epoch in list(map(lambda x: x+1,range(NUM_EPOCH))):
        print(epoch)
        for index in range(num_batches):
            X_d_true = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            X_g = np.array([np.random.normal(0,0.5,100) for _ in range(BATCH_SIZE)])
            X_d_gen = g.predict(X_g, verbose=0)
            y_d_true = np.reshape(np.array(y_d_true),(BATCH_SIZE,1))
            y_d_gen = np.reshape(np.array(y_d_gen),(BATCH_SIZE,1))
            y_g = np.reshape(np.array(y_g),(BATCH_SIZE,1))
            # train discriminator
            d_loss = d.train_on_batch(X_d_true, y_d_true)
            d_loss = d.train_on_batch(X_d_gen, y_d_gen)
            # train generator
            g_loss = dcgan.train_on_batch(X_g, y_g)
            #show_progress(epoch,index,g_loss[0],d_loss[0],g_loss[1],d_loss[1])
            arr = g.predict(seed)
            # print ("shape of arr: ",arr.shape)
            #res = generateImage(arr, data_shape)
            # #cv2.imshow('Generated Image', res) # on windows, i had to install xming
            # # print ("showing image")
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     sys.exit(0)
            # print ("drawn")
            dLosses.append(d_loss)
            gLosses.append(g_loss)
            #accuracies.append(accuracy)

        plot_every = 50
        if epoch % plot_every == 0:
            #arr = generator.predict(seed)
            plotLoss(epoch)
        saveAlbum(g, epoch, data_shape, z_pred)

        # save generated images
        # image = combine_images(g.predict(z_pred))
        # #image = image*127.5 + 127.5
        # Image.fromarray(image.astype(np.uint8))\
        #    .save(GENERATED_IMAGE_PATH+"%03depoch.png" % (epoch))
        # print()
        # # save models
        # g.save(GENERATED_MODEL_PATH+'dcgan_generator.hdf5')
        # d.save(GENERATED_MODEL_PATH+'dcgan_discriminator.hdf5')

def visualizeOne():
    arr = generator.predict(seed)
    # print ("shape of arr: ",arr.shape)
    res = generateImage(arr)
    cv2.imshow('Generated Image', res) # on windows, i had to install xming
    # print ("showing image")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)
    # print ("drawn")
# time.sleep(.1)

if __name__ == '__main__':
    train()
