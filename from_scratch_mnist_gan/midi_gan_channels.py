from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose
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
import math
from music21 import midi, stream, pitch, note, tempo, chord, instrument
os.environ["KERAS_BACKEND"] = "tensorflow"
np.set_printoptions(threshold=np.nan)

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
    parser.add_argument('--output', type=str, default='omni_generated',
                        help='directory of output (within colors)')
    parser.add_argument('--plot-every', type=int, default=20,
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

args = parse_args()

# data_source = "color/" + args.data + "_sprites"
# output_dir = "color/" + args.data + "_output"

data_source = "data/" + args.input
output_dir = "output/" + args.output


#change this directory to where hdf5 file is stored
DATASETS_DIR = os.path.dirname(os.path.realpath(__file__))

def loadMNIST(dataType):
    if(data_source == "data/mnist.hdf5"):
      imageDim = 28
    data_shape = (imageDim, imageDim, 1)
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

    return X, data_shape

def getImageDim():
    try:
        files = os.listdir(data_source)
    except:
        print ("cannot load directory: " + data_source)
        sys.exit(0)
    height, width, channels = cv2.imread(os.path.join(data_source,files[0])).shape
    print ("height: ", height, " width: ", width, " channels: ", channels)
    #returns height of first image
    return height

def loadPixels():
    channels = 3
    imageDim = getImageDim()
    data_shape = (imageDim, imageDim, channels)

    files = os.listdir(data_source)
    count = len(files)
    images = np.empty((count, imageDim, imageDim, channels))
    for i in range(count):
        pic = cv2.imread(os.path.join(data_source,files[i]))

        images[i] = pic
        if args.display:
            visualize(pic)
    return images/255, data_shape
    #return images
    #return np.reshape(images ,(count, imageDim**2 * 3))/255

# music stuff
lowest_pitch = 30
highest_pitch = 127
note_range = highest_pitch-lowest_pitch
beats_per_minisong = 16
instrument_list = []
MAX_VOL = 255
# LENGTH PER BEAT IS THE STANDARDIZED LENGTH OF NOTES/RESTS
# IT IS USED IN THE CALCULATION OF HOW MANY SONGS WE CAN CREATE
# IT EFFECTIVELY DEFINES THE MEASURE LENGTH
lengthPerBeat = 0.25
song_tempo = 100

def get_standardized_note_tracks(tracks, longest_track):
  final_tracks = np.zeros((longest_track, note_range, len(tracks)))
  t = 0
  # for each track
  for part in tracks:
    notes = part.flat.notesAndRests.stream()
    n = 0
    inst = part.getInstrument().instrumentName
    if(inst == None):
        continue
        print("NO INSTRUMENT")
    # for all the notes in the track including rests
    for note in notes:
      # for the beat in the measure as defined by the standard note length
      # print("note length", note.quarterLength)
      # print("number of beats in this note", int(note.quarterLength/lengthPerBeat))
      for which_beat in range(int(note.quarterLength/lengthPerBeat)):
        # handle both chords and notes - rests are left as zeros since we initialize with np.zeros
        if note.isChord:
            # print(n)
            for p in note.pitches:
                # put the pitch into the corresponding index in the array
                final_tracks[n][p.midi-lowest_pitch][t] = note.volume.velocity/MAX_VOL
        elif not note.isRest:
            # print(n)

            # put the pitch into the corresponding index in the array
            final_tracks[n][note.pitch.midi-lowest_pitch][t] = note.volume.velocity/MAX_VOL
        # next note
        n +=1
    # next track
    t += 1
  return final_tracks

def loadMidi():
    mf = midi.MidiFile()
    mf.open(filename = data_source)
    mf.read()
    mf.close()

    #read to stream
    s = midi.translate.midiFileToStream(mf)
    metronome = s.metronomeMarkBoundaries()[0]
    temp = metronome[2].getQuarterBPM()
    global song_tempo
    # set the tempo of the song to match it when we remidify
    song_tempo = temp

    #number of parts/instruments
    tracks = s.parts
    channels = len(tracks)
    data_shape = (beats_per_minisong, note_range, channels)

    # number of possible songs in the longest track
    num_songs = 0
    for track in tracks:
        inst = track.getInstrument()
        inst_name = inst.instrumentName
        # print("notes: ")
        # notes.show('text')
        if(inst_name == None):
            inst = instrument.fromString('Piano')
        global instrument_list
        # add our instrument to the array to keep track of instruments on each channel
        instrument_list.append(inst)
        # print("track duration quarterlength ", track.duration.quarterLength)
        total_beats = (math.ceil(track.duration.quarterLength)/lengthPerBeat)
        # print("total_beats :", total_beats)
        length = total_beats//beats_per_minisong
        # print("length :", length)
        if( length > num_songs):
            longest_track = track
            num_songs = int(length)
            # print("num_songs : ", num_songs)

    # Get back to length of song in 16th notes
    longest_track_length = num_songs*beats_per_minisong

    # get standarized tracks
    standardized_tracks = get_standardized_note_tracks(tracks, longest_track_length)

    # reshape to break them into "measures" as defined by beats_per_minisong
    minisongs = np.reshape(standardized_tracks, ((num_songs,) + data_shape)  )

    return minisongs, data_shape, num_songs

def reMIDIfy(minisong, output):
    # each note
    s1 = stream.Stream()
    # assign the tempo based on what was read in
    t = tempo.MetronomeMark('fast', song_tempo, note.Note(type='quarter'))
    # t = tempo.MetronomeMark('fast', 240, note.Note(type='quarter'))

    s1.append(t)
    minisong = minisong.reshape((beats_per_minisong, note_range, channels))
    for curr_channel in range(channels):
        new_part = stream.Part([instrument_list[curr_channel]])
        for beat in range(beats_per_minisong):
            notes = []
            for curr_pitch in range(note_range):
                #if this pitch is produced with at least 10% likelihood then count it
                if minisong[beat][curr_pitch][curr_channel]>.1:
                    p = pitch.Pitch()
                    p.midi = curr_pitch+lowest_pitch
                    n = note.Note(pitch = p)
                    n.pitch = p
                    n.volume.velocity = minisong[beat][curr_pitch][curr_channel]*MAX_VOL
                    n.quarterLength = lengthPerBeat
                    notes.append(n)
            if notes:
                my_chord = chord.Chord(notes)
            
            else:
                my_chord = note.Rest()
                my_chord.quarterLength = lengthPerBeat

            new_part.append(my_chord)
        s1.insert(curr_channel, new_part)

    mf = midi.translate.streamToMidiFile(s1)
    mf.open(output + ".mid", 'wb')
    mf.write()
    mf.close()

def writeCutSongs(notesData):

    directory = "output/midi_input"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # print ("notes data is: ", len(notesData))
    for x in range(len(notesData)):
        reMIDIfy(notesData[x], directory+"/input_song_"+str(x))
        # cv2.imwrite(directory+"/input_score_%d.png" % x, notesData[x]*255)

def loadData():
    if(data_source[-6:] == ".hdf5"):
        print (" MNIST! ")
        return loadMNIST("train")
    if data_source[-4:] == ".mid":
        print (" MUSIC! ")
        # song = loadMidi()
        # writeCutSongs(song[0])
        return loadMidi()
    else:
        print (" COLOR IMAGES! ")
        return loadPixels()

x_train, data_shape, num_songs = loadData() #grabbing all training inputs
channels = data_shape[2]
# print ("channels is: ", channels)
data_size = data_shape[0]*data_shape[1]*data_shape[2]


#defining noise vector size
noise_vect_size = 10
np.random.seed(1000)


# Optimizer
adam = Adam(lr=0.00001, beta_1=0.5)

#seed= np.random.rand(noise_vect_size)
seed = np.random.normal(0, 1, size=[1, noise_vect_size])


#testing sequential model
generator = Sequential()


#data_shape = (imageDim, imageDim, channels)


#stacking layers on model

generator.add(Dense(data_size, activation = 'sigmoid', input_dim=noise_vect_size, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
# generator.add(Dropout(.1))
generator.add(Reshape((data_shape), input_shape=(data_size,)))
generator.add(Conv2DTranspose(64, (2, 24), padding='same'))
generator.add(Conv2DTranspose(channels, (2, 24), padding='same'))

#compiling loss function and optimizer
generator.compile(loss = 'mse', optimizer = adam)

#create discriminator
discriminator = Sequential()
discriminator.add(Conv2D(32, (2, 24), padding='same', input_shape=(data_shape)))
# discriminator.add(MaxPooling2D(pool_size=(2, 2)))
discriminator.add(Dropout(.25))
discriminator.add(Conv2D(64, (2, 24), padding='same'))
# discriminator.add(MaxPooling2D(pool_size=(2, 2)))
discriminator.add(Dropout(.25))
discriminator.add(Flatten())

discriminator.add(Dense(32, activation = 'sigmoid', input_dim=data_size, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
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



def saveMidi(notesData, epoch):
    f = output_dir+"/song_"+str(epoch)
    reMIDIfy(notesData[0], f)
    # print (" saving song as ", f)




#end of music

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir + '/gan_loss_epoch_%d.png' % epoch)
    plt.close()
    # print ("Saving loss graph as "+ output_dir + "/gan_loss_epoch_%d.png" % epoch)

magnification = 10

def printIntro():
    print("input from: ", data_source, " output to: ", output_dir)
    print("batch size: ", batch_size, " epochs: ", epochs)

def generateImage(arr):
    magnification = 10
    img = arr
    res = cv2.resize(img, None, fx=magnification, fy=magnification, interpolation = cv2.INTER_NEAREST)
    return res

def visualize(arr):
    res = generateImage(arr[0])
    cv2.imshow('Generated Image', res) # on windows, i had to install xming
    if cv2.waitKey(1) & 0xFF == ord('q'):
        sys.exit(0)

def saveImage(arr, e, low_loss=False):
  img = generateImage(arr[0])*255
  if low_loss:
    cv2.imwrite(output_dir + '/low_loss_generated_image_epoch_%d.png' % e, img)
  else:
    cv2.imwrite(output_dir + '/generated_image_epoch_%d.png' % e, img)

#save a bunch of random images
def saveAlbum(e, shape = (3,3)):
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

def saveSummary():
    file = open(output_dir + "/description.txt","w")
    file.write("generator:")
    file.write(generator.to_yaml())
    file.write("discriminator:")
    file.write(discriminator.to_yaml())
    file.write("input from: "+ data_source)
    file.write("batch size: "+ str(batch_size) + " epochs: " + str(epochs))
    file.close()

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
            data_x = np.array([train_data[i] for i in chosen_data_indexes]) #get next batch of the right size from training data

            #train discriminator
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
                arr = generator.predict(seed)
                visualize(arr)

        # if e % 20 == 0 and e != 0:
        #     new_learning_rate -= 0.00001
        #     print("NEW LEARNING RATE IS: ", new_learning_rate)
        #     adam = Adam(lr=new_learning_rate, beta_1=0.5)
        #     gan.compile(loss = 'binary_crossentropy', optimizer = 'adam')


        dLosses.append(dloss)
        gLosses.append(gloss)
        print("Discriminator loss: ", dloss)
        print("Generator loss: ", gloss)

        if e % args.save_every == 0:
             # saveModels(e)
             arr = generator.predict(seed)
             # print ("arr.shape is:", arr.shape)
             if arr.shape == (1, beats_per_minisong, note_range, channels):
                 saveMidi(arr, e)
             # saveImage(arr, e)
        if e % args.plot_every == 0:
            arr = generator.predict(seed)
            plotLoss(e)

    plotLoss(e)

    return



if __name__ == '__main__':
    epochs = args.epochs
    batch_size = args.batch
    printIntro()
    saveSummary()
    print(x_train.shape)
    # test encoding by uncommenting below:
    # writeCutSongs(x_train[:num_songs])
    trainGAN(x_train, epochs = epochs, batch_size=batch_size) #begin training
