import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
# from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import sys
import cv2

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

K.set_image_dim_ordering('th')

# Deterministic output.
# Tired of seeing the same results every time? Remove the line below.
np.random.seed(1000)

# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim = 100

#imageDim = 28
imageDim = 31

input_dir = "fungi_sprites"
output_dir = "fungi_generated"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


X = 60
Y = 32

# Load MNIST data
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = (X_train.astype(np.float32) - 127.5)/127.5
# X_train = X_train.reshape(60000, 784)
# # TAKE SMALLER DATASET
# X_train = X_train[:10000]


def loadFaces():
    size = 2429
    #size = 10000
    f = h5py.File("faces.hdf5", 'r')
    X = f['data'][:size]
    return X

def loadPixels():
    my_dir = input_dir
    try:
        files = os.listdir(my_dir)
    except:
        print ("cannot load directory: " + my_dir)
        sys.exit(0)

    print ("reading file with dimensions: ", cv2.imread(os.path.join(my_dir,files[0])).shape)
    height, width, channels = cv2.imread(os.path.join(my_dir,files[0])).shape

    print ("channels: ", channels)

    #images = np.empty((Y*X, height, width, channels))
    images = np.empty((Y*X, height, width, 3))
    for i in range(len(files)):
        pic = cv2.imread(os.path.join(my_dir,files[i]))

        images[i] = pic
        visualizeTest(pic)
    return np.reshape(images ,(Y*X, imageDim**2 * 3))

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

generator = Sequential()
generator.add(Dense(200, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(Dense(200))
generator.add(Dense(200))
generator.add(Dense(imageDim**2*3, activation='sigmoid'))
generator.compile(loss='mse', optimizer=adam)

discriminator = Sequential()
discriminator.add(Dense(35, input_dim=imageDim**2*3, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='mse', optimizer=adam)

# Combined network
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

dLosses = []
gLosses = []

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir +'/gan_loss_epoch_%d.png' % epoch)
    plt.close();
    print ("Saving loss graph as " + output_dir + "/gan_loss_epoch_%d.png" % epoch)

# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, imageDim, imageDim, 3)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    print("**********saving")
    plt.savefig(output_dir + '/gan_generated_image_epoch_%d.png' % epoch)
    plt.close();

# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models/gan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/gan_discriminator_epoch_%d.h5' % epoch)

def train(X_train, epochs=1, batchSize=128):
    batchCount = X_train.shape[0] / batchSize
    print ('Epochs:', epochs)
    print ('Batch size:', batchSize)
    print ('Batches per epoch:', batchCount)

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for batch in range(int(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # Generate fake MNIST images, flattened with scale of 3 x scale
            generatedImages = generator.predict(noise)

            #print ("image batch shape ", np.shape(imageBatch), " generated image shape ", np.shape(generatedImages))
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)
            visualizeOne()

        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)
        print("Discriminator loss: ", dloss)
        print("Generator loss: ", gloss)


        if e % 20 == 0:
            plotGeneratedImages(e)
            #saveModels(e)

    # Plot losses from every epoch
    plotLoss(e)

#seed= np.random.rand(noise_vect_size)
seed = np.random.normal(0, 1, size=[1, randomDim])
print ("seed: ", seed.shape)

def generateImage(arr):
    magnification = 10
    img = np.reshape(arr, (imageDim, imageDim, 3))
    res = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
    return res

def visualizeOne():
    arr = generator.predict(seed)
    res = generateImage(arr)
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


if __name__ == '__main__':
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    #X_train = loadMNIST("train")

    X_train = loadPixels()
    train(X_train, epochs, batch_size)
