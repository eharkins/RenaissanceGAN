# Simple GAN implementation with keras
# adaptation of https://gist.github.com/Newmu/4ee0a712454480df5ee3
import sys, os

os.environ["KERAS_BACKEND"] = "tensorflow"

from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.optimizers import SGD
from keras.initializers import normal

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from scipy.io import wavfile

from music21 import midi, stream, pitch, note, tempo




batch_size = 32*32

print ("loading data")


#load midi file

mf = midi.MidiFile()
mf.open(filename = "bach.mid")
mf.read()
mf.close()

#read to stream
print ("reading to stream")

s = midi.translate.midiFileToStream(mf)

print ("converting to notes")

#convert to notes
notes = s.flat.notes
# print ("notes are: ", notes)
print ("notes length is: ", len(notes))

print ("converting back")
#convert back
s1 = stream.Stream(notes)

print ("notes2 length is: ", len(s1.flat.notes))

#write to file
mf = midi.translate.streamToMidiFile(s1)

print ("writing to file")
fname = "DATA_SONG.mid"

mf.open(fname, 'wb')
mf.write()
mf.close()
print ("saved as: ", fname)

#number of notes in each data example
minisong_size = 30

data_size = 3

num_songs = int(len(notes)/minisong_size)


minisongs = np.empty((num_songs, minisong_size*data_size))

#convert to 3d array of x by n by 5
#normalize notes

#notes is tuple of notes


for i in range(num_songs):
    for j in range(minisong_size):
        #print ("minisong: {} note: {}".format(i, notes[i*minisong_size+j]))
        # print ("length of array is: {} minisong: {} note:{} note value:{}".format(notes.size,i,j, i*minisong_size+j))
        n = notes[i*minisong_size+j]
        print ("notes is: ", notes)
        print ("n is: ", n, notes[1], notes[2], notes[300])

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
        n.quarterLength = byteSafe(notes[j+1])
        n.volume.velocity = byteSafe(notes[j+2]*100)
        print ("note {} is: {} ".format(j, note))
        s1.append(n)

    mf = midi.translate.streamToMidiFile(s1)
    mf.open(output + ".mid", 'wb')
    mf.write()
    mf.close()


#printing original song in small pieces
# directory = "test_output"
# if not os.path.exists(directory):
#     os.makedirs(directory)
# for i in range(num_songs):
#     print ("reMIDIfication time!")
#     reMIDIfy(minisongs[i], directory+"/song_fragment_"+str(i))

data = minisongs.astype(np.float64).reshape((-1,2))
print (data.shape)
data = data[:,0]+data[:,1]
data -= data.min()
data /= data.max() / 2.
data -= 1.
print (data.shape)

input_size = 2048

print ("Setting up decoder")
decoder = Sequential()
decoder.add(Dense(512, input_dim=input_size, activation='relu'))
decoder.add(Dropout(0.5))
decoder.add(Dense(512, activation='relu'))
decoder.add(Dropout(0.5))
decoder.add(Dense(1, activation='sigmoid'))

decoder.trainable=False

sgd = SGD(lr=0.01, momentum=0.1)
decoder.compile(loss='binary_crossentropy', optimizer=sgd)

print ("Setting up generator")
generator = Sequential()
generator.add(Dense(512*2, input_dim=512, activation='relu'))
generator.add(Dense(128*8, activation='relu'))
generator.add(Dense(input_size, activation='linear'))

generator.compile(loss='binary_crossentropy', optimizer=sgd)

print ("Setting up combined net")
gen_dec = Sequential()
gen_dec.add(generator)
gen_dec.add(decoder)

gen_dec.compile(loss='binary_crossentropy', optimizer=sgd)

y_decode = np.ones(2*batch_size)
y_decode[:batch_size] = 0.
y_gen_dec = np.ones(batch_size)

def gaussian_likelihood(X, u=0., s=1.):
    return (1./(s*np.sqrt(2*np.pi)))*np.exp(-(((X - u)**2)/(2*s**2)))


fig = plt.figure()



directory = "real_output"
if not os.path.exists(directory):
    os.makedirs(directory)

print ("number of examples is: ", data.shape)


#for i in range(100000):
for i in range(100):
    zmb = np.random.uniform(-1, 1, size=(batch_size, 512)).astype('float32')
    #xmb = np.random.normal(1., 1, size=(batch_size, 1)).astype('float32')
    xmb = np.array([data[n:n+input_size] for n in np.random.randint(0,data.shape[0]-input_size,batch_size)])
    if i % 10 == 0:
        r = gen_dec.fit(zmb,y_gen_dec,epochs=1,verbose=0)
        #print 'E:',np.exp(gen_dec.totals['loss']/batch_size)
        print (i ,' E Loss: ', gen_dec.losses)
    else:
        r = decoder.fit(np.vstack([generator.predict(zmb),xmb]),y_decode,epochs=1,verbose=0)
        #print 'D:',np.exp(gen_dec.totals['loss']/batch_size)
        print (i, ' D Loss: ', gen_dec.losses)
    if i % 10 == 0:
        print ("saving fakes")
        fakes = generator.predict(zmb[:16,:])
        for n in range(16):
            reMIDIfy(fakes[n,:], directory+"/fake_"+str(n))
            reMIDIfy(fakes[n,:], directory+"/real_"+str(n))
        vis(i)
