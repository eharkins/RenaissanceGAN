import os, cv2, math
from music21 import midi, stream, pitch, note, tempo, chord, instrument
import numpy as np

# music constants
LOWEST_PITCH = 30
HIGHEST_PITCH = 127
NOTE_RANGE = HIGHEST_PITCH-LOWEST_PITCH
BEATS_PER_MEASURE = 16
MEASURES_PER_MINISONG = 8
BEATS_PER_MINISONG = BEATS_PER_MEASURE * MEASURES_PER_MINISONG
MAX_VOL = 127
VOLUME_CUTOFF = 0.1
# LENGTH PER BEAT IS THE STANDARDIZED LENGTH OF NOTES/RESTS
# IT IS USED IN THE CALCULATION OF HOW MANY SONGS WE CAN CREATE
# IT EFFECTIVELY DEFINES THE MEASURE LENGTH
LENGTH_PER_BEAT = 0.25

song_tempo = 100

instrument_list = []
# put the pitches into the corresponding index in the array
def addNote(notes, final_tracks, measure_in_song, minisong, track_n):
    for note in notes:
        position = int(note.offset/LENGTH_PER_BEAT) + measure_in_song * BEATS_PER_MEASURE
        if note.isChord:
          for p in note.pitches:
            final_tracks[minisong, position, p.midi-LOWEST_PITCH, track_n] = 1 #note.volume.velocity/MAX_VOL
        elif not note.isRest:
            final_tracks[minisong, position, note.pitch.midi-LOWEST_PITCH, track_n] = 1 #note.volume.velocity/MAX_VOL


def get_standardized_note_tracks(num_songs, beats_per_minisong, tracks):
    #first dimension is minisong number * notes per minisong
    final_tracks = np.zeros((num_songs, beats_per_minisong, NOTE_RANGE, len(tracks)))
    print ("tracks size is: ", len(tracks))
    print ("final tracks shape is: ", final_tracks.shape)
    for track_n in range(len(tracks)):
        track = tracks[track_n]
        # add our instrument to the array to keep track of instruments on each channel
        inst = track.getInstrument()
        inst_name = inst.instrumentName
        print ("instrument", track_n, " is: ", inst_name)
        global instrument_list
        if(inst_name == None):
            print("NO INSTRUMENT")
            inst_name = 'Piano'
            instrument_list.append(inst_name)
            continue
        instrument_list.append(inst_name)

        measures = track.flat.notes.stream().measures(0, None)
        measures = measures.getElementsByClass("Measure")
        print ("number of measures: ", len(measures))
        for measure in range(len(measures)):
            m = measures[measure]
            minisong = int(measure/MEASURES_PER_MINISONG)
            measure_in_song = measure%MEASURES_PER_MINISONG
            if m.voices:
                for v in m.voices:
                    addNote(v.notes, final_tracks, measure_in_song, minisong, track_n)
            else:
                addNote(m.notes, final_tracks, measure_in_song, minisong, track_n)
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
    # preserve the tempo
    song_tempo = temp

    #number of parts/instruments
    tracks = s.parts
    #tracks = tracks[:3]
    print("CHANNELS : ", len(tracks))
    channels = len(tracks)
    data_shape = (BEATS_PER_MINISONG, NOTE_RANGE, channels)

    # number of possible songs in the longest track
    longest_length = 0
    for track in tracks:
        print("track length: ", track.duration.quarterLength)
        longest_length = max(longest_length, track.duration.quarterLength)
    mybeats = longest_length/LENGTH_PER_BEAT
    num_songs = math.ceil(mybeats/BEATS_PER_MINISONG)
    minisongs = get_standardized_note_tracks(num_songs, BEATS_PER_MINISONG, tracks)
    return minisongs, data_shape

def reMIDIfy(minisong, output):
    # each note
    s1 = stream.Stream()
    # assign the tempo based on what was read
    t = tempo.MetronomeMark('fast', song_tempo, note.Note(type='quarter'))
    s1.append(t)
    channels = minisong.shape[2]
    for curr_channel in range(channels):
        inst = instrument.fromString(instrument_list[curr_channel])
        new_part = stream.Part()
        new_part.insert(0, inst)
        print(instrument_list[curr_channel])
        for beat in range(BEATS_PER_MINISONG):
            notes = []
            for curr_pitch in range(NOTE_RANGE):
                #if this pitch is produced with at least 10% likelihood then count it
                if minisong[beat][curr_pitch][curr_channel]>VOLUME_CUTOFF:
                    p = pitch.Pitch()
                    p.midi = curr_pitch+LOWEST_PITCH
                    n = note.Note(pitch = p)
                    n.pitch = p
                    n.volume.velocity = minisong[beat][curr_pitch][curr_channel]*MAX_VOL
                    n.quarterLength = LENGTH_PER_BEAT
                    notes.append(n)
            if notes:
                my_chord = chord.Chord(notes)

            else:
                my_chord = note.Rest()
                my_chord.quarterLength = LENGTH_PER_BEAT

            new_part.append(my_chord)
        s1.insert(curr_channel, new_part)

    mf = midi.translate.streamToMidiFile(s1)
    mf.open(output + ".mid", 'wb')
    mf.write()
    mf.close()

def playSong(music_file):
    #clock = pygame.time.Clock()
    try:
        pygame.mixer.music.load(music_file)
        print ("playing song:", music_file)
    except pygame.error:
        print ("File %s not found! (%s)" % (music_file, pygame.get_error()))
        return
    except KeyboardInterrupt:
        # if user hits Ctrl/C then exit
        # (works only in console mode)
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit
    pygame.mixer.music.play()
    # while pygame.mixer.music.get_busy():
    #     # check if playback has finished
    #     clock.tick(30)



def saveMidi(notesData, epoch, output_dir):
    f = output_dir+"/song_"+str(epoch)
    reMIDIfy(notesData[0], f)
    # if not pygame.mixer.music.get_busy():
    #     playSong(f)

def writeCutSongs(notesData, directory = "output/midi_input"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for x in range(len(notesData)):
        reMIDIfy(notesData[x], directory+"/input_song_"+str(x))
