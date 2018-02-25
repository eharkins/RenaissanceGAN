#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
from keras.datasets import mnist
from PIL import Image
from model import discriminator, generator
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from visualizer import *

DATASETS_DIR = os.path.dirname(os.path.realpath(__file__));
os.chdir(DATASETS_DIR) 

BATCH_SIZE = 32
NUM_EPOCH = 50
LR = 0.0002  # initial learning rate
B1 = 0.5  # momentum term
GENERATED_IMAGE_PATH = 'images/'
GENERATED_MODEL_PATH = 'models/'

magnification = 10
def generateImage(arr):
    img = np.reshape(arr, (28, 28))
    res = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
    return res

def train():
    (X_train, y_train), (_, _) = mnist.load_data()
    # normalize images- this is the line to look for
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    print(X_train)
    # build GAN
    g = generator()
    d = discriminator()

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
    # create directory
    if not os.path.exists(GENERATED_IMAGE_PATH):
        os.mkdir(GENERATED_IMAGE_PATH)
    if not os.path.exists(GENERATED_MODEL_PATH):
        os.mkdir(GENERATED_MODEL_PATH)

    print("-------------------")
    print("Total epoch:", NUM_EPOCH, "Number of batches:", num_batches)
    print("-------------------")
    z_pred = np.array([np.random.uniform(-1,1,100) for _ in range(49)])
    y_g = [1]*BATCH_SIZE
    y_d_true = [1]*BATCH_SIZE
    y_d_gen = [0]*BATCH_SIZE
    seed = np.random.normal(0, 1, size=[1, 100])
    for epoch in list(map(lambda x: x+1,range(NUM_EPOCH))):
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
            show_progress(epoch,index,g_loss[0],d_loss[0],g_loss[1],d_loss[1])
            arr = g.predict(seed)
            # print ("shape of arr: ",arr.shape)
            res = generateImage(arr)
            cv2.imshow('Generated Image', res) # on windows, i had to install xming
            # print ("showing image")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sys.exit(0)
            # print ("drawn")

        # save generated images
        image = combine_images(g.predict(z_pred))
        image = image*127.5 + 127.5
        Image.fromarray(image.astype(np.uint8))\
           .save(GENERATED_IMAGE_PATH+"%03depoch.png" % (epoch))
        print()
        # save models
        g.save(GENERATED_MODEL_PATH+'dcgan_generator.h5')
        d.save(GENERATED_MODEL_PATH+'dcgan_discriminator.h5')

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
