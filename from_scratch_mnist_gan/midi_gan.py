import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from music21 import midi, stream, pitch, note, tempo

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
from keras.utils import to_categorical

K.set_image_dim_ordering('th')

# Deterministic output.
# Tired of seeing the same results every time? Remove the line below.
np.random.seed(1000)

# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim = 100

# Number of notes in each data example
minisong_size = 8

# Number of data in each note: pitch, length, and veocity. Velocity is currently auto-set to 100.

#use pitches between 48 and 84 so note size is going to be 84-48+1 = 37
note_size = 37
data_size = minisong_size*note_size

arrpeggio = [48, 60, 72, 84, 48, 60, 72, 84]
arrpeggio[:] = [x - 48 for x in arrpeggio]
encoded = to_categorical(arrpeggio, note_size)
print(encoded)
print(encoded.shape)
encoded = encoded.reshape(data_size)
print(encoded)
X_train = np.zeros((1000, data_size))
for i in range(1000):
    X_train[i] = encoded

def loadMidi():
    mf = midi.MidiFile()
    mf.open(filename = "bach.mid")
    mf.read()
    mf.close()

    #read to stream
    s = midi.translate.midiFileToStream(mf)

    #convert to notes
    notes = s.flat.notes
    num_songs = int(len(notes)/minisong_size)
    minisongs = np.zeros((num_songs, minisong_size, note_size))

    for i in range(num_songs):
        for j in range(minisong_size):
            for k in range(note_size):

                #i don't know if thi gets multiple notes played at the same time / how this works
                if not n.isChord:
                    p = (n.pitch.midi)/37
                else:
                    p = (n.pitches[0].midi)/37
                minisongs[i][j][p] = 1

                print("pitch: ", p)
    return minisongs

def byteSafe(num):
    if (num < 0):
        return 0
    elif num > 256:
        return 256
    else: return num

def reMIDIfy(minisong, output):
    # each note
    s1 = stream.Stream()
    t = tempo.MetronomeMark('fast', 240, note.Note(type='quarter'))
    s1.append(t)
    minisong = minisong.reshape((minisong_size, note_size))
    print(minisong)
    for j in range(minisong_size):
        p = pitch.Pitch()
        p.midi = np.argmax(minisong[j])+48
        print(p)
        n = note.Note(pitch = p)
        n.pitch = p

        n.volume.velocity = 255
        n.quarterLength = 4

        s1.append(n)

    #add a rest at the end, hopefully this will make it longer
    r = note.Rest()
    r.quarterLength = 4
    s1.append(r)

    mf = midi.translate.streamToMidiFile(s1)
    mf.open(output + ".mid", 'wb')
    mf.write()
    mf.close()

def saveMidi(notesData, epoch):

    directory = "midi_output"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for x in range(len(notesData)):
        reMIDIfy(notesData[x], directory+"/song_"+str(epoch)+"_"+str(x))


def writeCutSongs(notesData):

    directory = "midi_input"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for x in range(len(notesData)):
        reMIDIfy(notesData[x], directory+"/input_song_"+str(x))

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

generator = Sequential()
generator.add(Dense(200, activation='sigmoid', input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(Dense(data_size, activation='sigmoid'))
generator.compile(loss='mse', optimizer=adam)

discriminator = Sequential()
discriminator.add(Dense(35, activation='sigmoid', input_dim=data_size, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='mse', optimizer=adam)

# Combined network
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='mse', optimizer=adam)

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
    plt.savefig('midi_output/gan_loss_epoch_%d.png' % epoch)
    print ("Saving loss graph as midi_output/gan_loss_epoch_%d.png" % epoch)


# Create a wall of images, use with Xtrain to display input data
def plotImages(images, file_name, examples=100, dim=(10, 10), figsize=(10, 10)):
    images = images[0:examples].reshape(examples, minisong_size, note_size)
    plt.figure(figsize=figsize)
    for i in range(images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    print("**********saving")
    plt.savefig(file_name+'.png')

# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)
    generatedImages = generatedImages.reshape(examples, minisong_size, note_size)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    print("**********saving")

    directory = "midi_output"
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig('midi_output/gan_generated_image_epoch_%d.png' % epoch)

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

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            # print np.shape(imageBatch), np.shape(generatedImages)
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

        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)
        print("Discriminator loss: ", dloss)
        print("Generator loss: ", gloss)

        if e == 1 or e % 5 == 0:
            # saveModels(e)
            arr = generator.predict(seed)
            saveMidi(arr, e)
            if e % 50 == 0:
                plotGeneratedImages(e)

    # Plot losses from every epoch
    plotLoss(e)

magnification = 10

#seed= np.random.rand(noise_vect_size)
seed = np.random.normal(0, 1, size=[1, randomDim])
print(X_train)
print(X_train.shape)
if __name__ == '__main__':
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    #X_train = loadMNIST("train")
    #X_train = loadMidi()
    #writeCutSongs(X_train)
    #plotImages(X_train, "midi_input/input_data")
    train(X_train, epochs, batch_size)
