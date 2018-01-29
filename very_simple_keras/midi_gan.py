# Simple GAN implementation with keras
# adaptation of https://gist.github.com/Newmu/4ee0a712454480df5ee3
import sys, os

os.environ["KERAS_BACKEND"] = "tensorflow"

# from keras.models import Sequential
# from keras.layers.core import Dense,Dropout
# from keras.optimizers import SGD
# from keras.initializers import normal

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from scipy.io import wavfile

import madmom.utils.midi as md


batch_size = 32*32

print ("loading data")


#load midi file

m = md.MIDIFile.from_file("mk.mid")

# n by 5 array for number of notes
notes = m.notes()

print ("m.notes is:")
print (notes)

#testing stuff here

song = md.MIDIFile.from_notes(notes)
print ("song notes is:")
print (song.notes())
song.write("output/whole_song_in_and_out.mid")


#number of notes in each data example
minisong_size = 4

num_songs = int(notes.shape[0]/minisong_size)


minisongs = np.empty((num_songs, minisong_size*5))

#convert to 3d array of x by n by 5
#normalize notes

# print ("num songs: %s" % num_songs)
# print ("minisongs size: {}".format(minisongs.shape))
# print ("notes size: {}".format(notes.shape[0]))

for i in range(num_songs):
    for j in range(minisong_size):
        #print ("minisong: {} note: {}".format(i, notes[i*minisong_size+j]))
        # print ("length of array is: {} minisong: {} note:{} note value:{}".format(notes.size,i,j, i*minisong_size+j))

        #start time
        minisongs[i][0+j] = notes[i*minisong_size+j][0]/7.29166667
        #pitch
        minisongs[i][1+j] = (notes[i*minisong_size+j][1]-21)/24
        #duration
        minisongs[i][2+j] = notes[i*minisong_size+j][2]/2.08333333
        #velocity
        minisongs[i][3+j] = notes[i*minisong_size+j][3]/100
        #channel
        minisongs[i][4+j] = notes[i*minisong_size+j][4]

def byteSafe(num):
    if (num < 0):
        return 0
    elif num > 256:
        return 256
    else: return num

def reMIDIfy(notes, output):
    # each note
    song = np.empty((minisong_size, 5))
    for j in range(int(minisong_size)):
        song[j][0] = byteSafe(notes[j+0]*7.29166667)
        song[j][1] = byteSafe(notes[j+1]*24 + 21)
        song[j][2] = byteSafe(notes[j+2]*2.08333333)
        song[j][3] = byteSafe(notes[j+3]*100)
        #song[j][4] = byteSafe(notes[j+4])
        #channel - 0-15
        song[j][4] = 1
        print ("note {} is: {} ".format(j, song[j]))
    m = md.MIDIFile.from_notes(song)
    m.write(output + ".mid")

directory = "output"
if not os.path.exists(directory):
    os.makedirs(directory)
for i in range(num_songs):
    print ("reMIDIfication time!")
    reMIDIfy(minisongs[i], directory+"/song_fragment_"+str(i))

data = minisongs.astype(np.float64).reshape((-1,2))
print (data.shape)
data = data[:,0]+data[:,1]
data -= data.min()
data /= data.max() / 2.
data -= 1.
print (data.shape)

print ("Setting up decoder")
decoder = Sequential()
decoder.add(Dense(512, input_dim=32768, activation='relu'))
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
generator.add(Dense(32768, activation='linear'))

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

#for i in range(100000):
for i in range(100):
    zmb = np.random.uniform(-1, 1, size=(batch_size, 512)).astype('float32')
    #xmb = np.random.normal(1., 1, size=(batch_size, 1)).astype('float32')
    xmb = np.array([data[n:n+32768] for n in np.random.randint(0,data.shape[0]-32768,batch_size)])
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
            wavfile.write('output/fake_'+str(n+1)+'.mid',44100,fakes[n,:])
            wavfile.write('output/real_'+str(n+1)+'.mid',44100,xmb[n,:])
#        vis(i)
