import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from music21 import midi, stream, pitch, note, tempo, chord

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
from keras.layers.convolutional import Convolution2D, UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose

np.set_printoptions(threshold=np.nan)

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
lowest_pitch = 30
highest_pitch = 84
note_size = highest_pitch-lowest_pitch
data_size = minisong_size*note_size

channels = 1

# arrpeggio = [48, 60, 72, 84, 48, 60, 72, 84]
# arrpeggio[:] = [x - 48 for x in arrpeggio]
# encoded = to_categorical(arrpeggio, note_size)
# encoded = encoded.reshape(data_size)
# encoded[12] = 1
# X_train = np.zeros((1000, data_size))
# for i in range(1000):
#     X_train[i] = encoded

def loadMidi():
    mf = midi.MidiFile()
    mf.open(filename = "data/bach.mid")
    mf.read()
    mf.close()

    #read to stream
    s = midi.translate.midiFileToStream(mf)

    #convert to notes
    notes = s.flat.notes

    # num_songs = int(len(notes)/minisong_size)
    # print(num_songs)
    num_songs = 60
    # print("number of minisongs:  ", num_songs)
    minisongs = np.zeros((num_songs, minisong_size, note_size))

###########################################
    for i in range(num_songs):
        for j in range(minisong_size):
            # for k in range(note_size):
            note = notes[i*minisong_size + j]
            #i don't know if thi gets multiple notes played at the same time / how this works
            if not note.isChord:
                minisongs[i][j][note.pitch.midi-lowest_pitch] = 1
            else:
                chord_notes = []
                for p in note.pitches:
                    # chord_notes.append(p.midi-48)
                    minisongs[i][j][p.midi-lowest_pitch] = 1

            # print("pitch: ", p)
    minisongs = minisongs.reshape((num_songs, data_size))
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

    for j in range(len(minisong)):
        c = []
        for i in range(len(minisong[0])):
            # print("loop iteration:  "  , i)
            #if this pitch is produced with at least 80% likelihood then count it
            if minisong[j][i]>.5:
                # print("should be a note")
                c.append(i+lowest_pitch)
                #look up music21 stuff;  These i values/indexes are the notes in a chord

        if(len(c) > 0):
            p = chord.Chord(c)
            eventlist = midi.translate.chordToMidiEvents(p)
            p.volume.velocity = 255
            p.quarterLength = 1
        else:
            p = note.Rest()
        # elif(len(c) ==1):
        #     p = pitch.Pitch()
        #     p.midi = c[0]
        #     print(p.midi)
        # else:
        #     n = n.Rest()

        # n = note.Note(pitch = p)
        # n.pitch = p



        s1.append(p)

    #add a rest at the end, hopefully this will make it longer
    r = note.Rest()
    r.quarterLength = 4
    s1.append(r)

    mf = midi.translate.streamToMidiFile(s1)
    mf.open(output + ".mid", 'wb')
    mf.write()
    mf.close()

def saveMidi(notesData, epoch):

    directory = "midi_output_channels_test"
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



minisong_shape = (minisong_size, note_size, channels)

#testing sequential model
generator = Sequential()

#stacking layers on model
generator.add(Dense(35, activation = 'sigmoid', input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(Dropout(.1))
generator.add(Dense(data_size, activation = 'sigmoid'))
generator.add(Dropout(.1))
generator.add(Reshape(minisong_shape))
Conv2DTranspose(35, (3,24), padding='same', activation='sigmoid', data_format="channels_last")
Conv2DTranspose(1, (3,24), padding='same', activation='sigmoid', data_format="channels_last")
#generator.add(Flatten())

#compiling loss function and optimizer
generator.compile(loss = 'mse', optimizer = adam)

#create discriminator
discriminator = Sequential()
#discriminator.add(Reshape((imageDim, imageDim, 3), input_shape=(imageDim**2*3,)))
discriminator.add(Conv2D(35, (3, 24), padding='same', activation = 'sigmoid', input_shape=(minisong_shape), data_format="channels_last"))
# discriminator.add(MaxPooling2D(pool_size=(2, 2)))
discriminator.add(Conv2D(35, (3, 24), padding='same', activation = 'sigmoid', data_format="channels_last"))
# discriminator.add(MaxPooling2D(pool_size=(2, 2)))
discriminator.add(Flatten())

discriminator.add(Dense(35, activation = 'sigmoid', input_dim=data_size*channels, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(Dense(1, activation = 'sigmoid'))

#compiling loss function and optimizer
discriminator.compile(loss = 'mse', optimizer = adam)

# Combined network
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
# print(x)
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
    plt.savefig('midi_output_channels_test/gan_loss_epoch_%d.png' % epoch)
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

    directory = "midi_output_channels_test"
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig('midi_output_channels_test/gan_generated_image_epoch_%d.png' % epoch)

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
            imageBatch= np.reshape(imageBatch, (batch_size, minisong_size, note_size, channels))

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            # print("generatedImages size   :        ", generatedImages.shape)
            # print np.shape(imageBatch), np.shape(generatedImages)
            # print("imagebatch size:  ", imageBatch.shape)
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

        if e == 1 or e % 50 == 0:
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
# print(X_train)
# print(X_train.shape)
if __name__ == '__main__':
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    #X_train = loadMNIST("train")
    X_train = loadMidi()
    # reMIDIfy(X_train[1], "midi_output/test")
    #writeCutSongs(X_train)
    #plotImages(X_train, "midi_input/input_data")
    train(X_train, epochs, batch_size)
