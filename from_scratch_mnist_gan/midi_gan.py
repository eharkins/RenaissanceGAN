import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
# from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import h5py
import sys
#import cv2
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

K.set_image_dim_ordering('th')

# Deterministic output.
# Tired of seeing the same results every time? Remove the line below.
np.random.seed(1000)

# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim = 100


# Number of notes in each data example
minisong_size = 9

# Number of data in each note: pitch, length, and veocity. Velocity is currently auto-set to 100.
note_size = 3

data_size = note_size * minisong_size



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
    minisongs = np.empty((num_songs, data_size))

    for i in range(num_songs):
        for j in range(minisong_size):
            #print ("minisong: {} note: {}".format(i, notes[i*minisong_size+j]))
            # print ("length of array is: {} minisong: {} note:{} note value:{}".format(notes.size,i,j, i*minisong_size+j))
            n = notes[i*minisong_size+j]
            # print ("notes is: ", notes)
            # print ("n is: ", n, notes[1], notes[2], notes[300])

            # #start time
            # minisongs[i][0+j] = note.pitch.midi/7.29166667
            #
            #pitch
            if not n.isChord:
                p = (n.pitch.midi-21)/24
            else:
                p = (n.pitches[0].midi-21/24)
            minisongs[i][0+j] = p

            #duration
            #n2.duration
            minisongs[i][1+j] = n.quarterLength

            #velocity
            #n2.volume

            minisongs[i][3+j] = n.volume.velocity/100
            # #channel
            # minisongs[i][4+j] = notes[i*minisong_size+j][4]

        return minisongs


def byteSafe(num):
    if (num < 0):
        return 0
    elif num > 256:
        return 256
    else: return num

def reMIDIfy(notes, output):
    # each note
    s1 = stream.Stream()
    t = tempo.MetronomeMark('fast', 240, note.Note(type='quarter'))
    s1.append(t)
    song = np.empty((minisong_size, 5))
    for j in range(int(minisong_size)):

        p = pitch.Pitch()
        p.midi = byteSafe(notes[j+0]*24 + 21)
        n = note.Note(pitch = p)
        n.pitch = p
        n.quarterLength = byteSafe((int(notes[j+1]*8))/8)
        n.volume.velocity = byteSafe(notes[j+2]*100)
        #all maximum velocity
        n.volume.velocity = 255
        #print ("note {} is: {} ".format(j, note))
        s1.append(n)

    #add a rest at the end, hopefully this will make it longer
    r = note.Rest()
    r.quarterLength = 2.5
    s1.append(r)

    mf = midi.translate.streamToMidiFile(s1)
    mf.open(output + ".mid", 'wb')
    mf.write()
    mf.close()

def saveMidi(notesData, epoch):
    # print(notesData)
    # x = tf.placeholder(tf.float32, shape = (8, 10))
    # #notes = generator.eval(( feed_dict={self.z: self.sample_z}))
    # notes = generator.eval(feed_dict={x: np.array((8,10))},  tf.Session())
    #convert to numpy array
    #convert to midi
    # print(notesData)

    print(len(notesData))
    print(len(notesData[0]))
    directory = "midi_output"
    if not os.path.exists(directory):
        os.makedirs(directory)
    for x in range(len(notesData)):
        reMIDIfy(notesData[x], directory+"/song_"+str(epoch)+"_"+str(x))
        # songNotes = batchToNotes(notesData[x])
        # m = md.MIDIFile.from_notes(songNotes)
        # # print(songNotes)
        # #save to file
        # filename = str(x)
        # m.write("song" + filename + ".mid")

# def loadMNIST(dataType):
#     #parameter determines whether data is training or testing
#     size = 10000
#     f = h5py.File("mnist.hdf5", 'r')
#     #f = h5py.File("faces.hdf5", 'r')
#     #X = f['data'][:size]
#     X = f['x_'+dataType][:size]
#
#     maxes = X.max(axis=0)
#     for i in range(len(maxes)):
#         if maxes[i] == 0:
#             maxes[i] = 0.1
#     X *= 1/maxes
#     # print X.shape
#
#     raw_y = np.array([f['t_'+dataType][:size]]).T
#     #raw_y = np.array([f['data'][:size]]).T
#
#     y = []
#     for row in raw_y:
#         #row 0 should be integer
#         y.append(convertToOneHot(row[0], 10))
#
#     y = np.array(y)
#
#     print ("MNIST Dataset LOADED")
#


# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

generator = Sequential()
generator.add(Dense(200, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(Dense(200))
generator.add(Dense(200))
generator.add(Dense(data_size, activation='sigmoid'))
generator.compile(loss='mse', optimizer=adam)

discriminator = Sequential()
discriminator.add(Dense(35, input_dim=data_size, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
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
    plt.savefig('midi_output/gan_loss_epoch_%d.png' % epoch)
    print ("Saving loss graph as images_original/gan_loss_epoch_%d.png" % epoch)


# Create a wall of images, use with Xtrain to display input data
def plotImages(images, examples=100, dim=(10, 10), figsize=(10, 10)):
    images = images[0:100].reshape(examples, minisong_size, note_size)
    plt.figure(figsize=figsize)
    for i in range(images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    print("**********saving")
    plt.savefig('midi_output/input_data_.png')




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
            plotGeneratedImages(e)
            # saveModels(e)
            arr = generator.predict(seed)
            saveMidi(arr, e)

    # Plot losses from every epoch
    plotLoss(e)

magnification = 10

#seed= np.random.rand(noise_vect_size)
seed = np.random.normal(0, 1, size=[1, randomDim])
print ("seed: ", seed.shape)

if __name__ == '__main__':
    epochs = int(sys.argv[1])
    batch_size = int(sys.argv[2])
    #X_train = loadMNIST("train")
    X_train = loadMidi()
    plotImages(X_train)
    train(X_train, epochs, batch_size)
