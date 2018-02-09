import os, cv2
from music21 import midi, stream, pitch, note, tempo, chord
import numpy as np

# music stuff
lowest_pitch = 30
highest_pitch = 127
note_range = highest_pitch-lowest_pitch
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
    global instrument_list
    # add our instrument to the array to keep track of instruments on each channel
    instrument_list.append(part.getInstrument())
    notes = part.flat.notesAndRests.stream()
    n = 0
    # for all the notes in the track including rests
    for note in notes:
      # for the beat in the measure as defined by the standard note length
      for which_beat in range(int(note.quarterLength/lengthPerBeat)):
        # handle both chords and notes - rests are left as zeros since we initialize with np.zeros
        if note.isChord:
          for p in note.pitches:
            # put the pitch into the corresponding index in the array
            final_tracks[n][p.midi-lowest_pitch][t] = note.volume.velocity/MAX_VOL
        elif not note.isRest:
            # put the pitch into the corresponding index in the array
            final_tracks[n][note.pitch.midi-lowest_pitch][t] = note.volume.velocity/MAX_VOL
        # next note
        n +=1
    # next track
    t += 1
  return final_tracks

def loadMidi(data_source):
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
      length = (track.duration.quarterLength/lengthPerBeat)//beats_per_minisong
      if( length > num_songs):
        num_songs = int(length)

    # Get back to length of song in 16th notes
    longest_track = num_songs*beats_per_minisong

    # get standarized tracks
    standardized_tracks = get_standardized_note_tracks(tracks, longest_track)

    # reshape to break them into "measures" as defined by beats_per_minisong
    minisongs = np.reshape(standardized_tracks, ((num_songs,) + data_shape)  )

    return minisongs, data_shape

def reMIDIfy(minisong, output):
    # each note
    s1 = stream.Stream()
    # assign the tempo based on what was read in
    t = tempo.MetronomeMark('fast', song_tempo, note.Note(type='quarter'))
    # t = tempo.MetronomeMark('fast', 240, note.Note(type='quarter'))
    s1.append(t)
    #minisong = minisong.reshape((beats_per_minisong, note_range, channels))
    channels = minisong.shape[2]
    #data_shape = (beats_per_minisong, note_range, channels)
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

def saveMidi(notesData, epoch, output_dir):
    f = output_dir+"/song_"+str(epoch)
    reMIDIfy(notesData[0], f)
    print (" saving song as ", f)


def writeCutSongs(notesData, output = "output"):

    directory = output + "/midi_input"
    if not os.path.exists(directory):
        os.makedirs(directory)
    #print ("number of song fragments: ", len(notesData))
    #print ("shape of notes is: ", notesData.shape)
    notesData = notesData[:,:,:,:3]
    for x in range(len(notesData)):
        reMIDIfy(notesData[x], directory+"/input_song_"+str(x))
        cv2.imwrite(directory+"/input_score_%d.png" % x, notesData[x]*255)
