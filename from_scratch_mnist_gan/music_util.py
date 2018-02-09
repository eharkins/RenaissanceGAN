import os, cv2
from music21 import midi, stream, pitch, note, tempo, chord
import numpy as np

# music stuff
lowest_pitch = 30
highest_pitch = 84
note_range = highest_pitch-lowest_pitch
minisong_size = 8

def loadMidi(data_source):
    # Number of notes in each data example

    #use pitches between 48 and 84 so note size is going to be 84-48+1 = 37


    data_shape = (minisong_size, note_range, 1)

    mf = midi.MidiFile()
    mf.open(filename = data_source)
    mf.read()
    mf.close()

    #read to stream
    s = midi.translate.midiFileToStream(mf)

    #convert to notes
    notes = s.flat.notes

    num_songs = int(len(notes)/minisong_size)
    # print("number of minisongs:  ", num_songs)
    minisongs = np.zeros(((num_songs,) + data_shape))

    for i in range(num_songs):
        for j in range(minisong_size):
            # for k in range(note_range):
            note = notes[i*minisong_size + j]
            # calvin doesn't know if thi gets multiple notes played at the same time / how this works
            if not note.isChord:
                minisongs[i][j][note.pitch.midi-lowest_pitch] = 1
            else:
                chord_notes = []
                for p in note.pitches:
                    # chord_notes.append(p.midi-48)
                    minisongs[i][j][p.midi-lowest_pitch] = 1

            # print("pitch: ", p)
    #minisongs = minisongs.reshape((num_songs, minisong_size*note_range))
    return minisongs, data_shape

def reMIDIfy(minisong, output):
    # each note
    s1 = stream.Stream()
    t = tempo.MetronomeMark('fast', 240, note.Note(type='quarter'))
    s1.append(t)
    #print ("Mininsong shape is: ", minisong.shape)
    minisong = minisong.reshape((minisong_size, note_range))
    #minisong = minisong[0]

    for j in range(len(minisong)):
        c = []
        for i in range(len(minisong[0])):
            #if this pitch is produced with at least 50% likelihood then count it
            if minisong[j][i]>.5:
                # print("should be a note")
                c.append(i+lowest_pitch)
                # i indexes are the notes in a chord

        if(len(c) > 0):
            n = chord.Chord(c)
            n.volume.velocity = 255
            n.quarterLength = 1
        # print ("c[0] is: ", c)
        # if(len(c) > 0):
        #     p = pitch.Pitch()
        #     p.midi= c[0] #testing with just 1 note
        #     n = note.Note(pitch = p)
        #     n.volume.velocity = 255
        #     n.quarterLength = 1
        else:
            n = note.Rest()
            n.quarterLength = 1

        #print ("chord is: ", p.pitches)
        s1.append(n)

    #add a rest at the end, hopefully this will make it longer
    r = note.Rest()
    r.quarterLength = 4
    s1.append(r)

    #print ("stream is: ", s1.flat.notes)
    #s1.append(p)

    mf = midi.translate.streamToMidiFile(s1)
    mf.open(output + ".mid", 'wb')
    mf.write()
    mf.close()

def saveMidi(notesData, epoch, output_dir = "output"):
    f = output_dir+"/song_"+str(epoch)
    reMIDIfy(notesData[0], f)
    print (" saving song as ", f)


def writeCutSongs(notesData, output = "output"):

    directory = output + "/midi_input"
    if not os.path.exists(directory):
        os.makedirs(directory)
    print ("number of song fragments: ", len(notesData))
    for x in range(len(notesData)):
        reMIDIfy(notesData[x], directory+"/input_song_"+str(x))
        cv2.imwrite(directory+"/input_score_%d.png" % x, notesData[x]*255)
