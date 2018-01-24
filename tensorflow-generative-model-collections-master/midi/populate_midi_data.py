import h5py
import numpy as np
import midi
import madmom.utils.midi as md
import random

#Generate lots of simple MIDI patterns containing tracks

arr = np.empty([600, 10])

for x in range(600):
  i = random.randint(21, 33)
  # Instantiate a MIDI Pattern (contains a list of tracks)
  pattern = midi.Pattern()
  # Instantiate a MIDI Track (contains a list of MIDI events)
  track = midi.Track()
  # Append the track to the pattern
  pattern.append(track)
  # Instantiate a MIDI note on event, append it to the track
  on = midi.NoteOnEvent(tick=1000, velocity=100, pitch=i)
  track.append(on)
  # Instantiate a MIDI note off event, append it to the track
  off = midi.NoteOffEvent(tick=2000, pitch=i)
  track.append(off)

  on = midi.NoteOnEvent(tick=1000, velocity=100, pitch=i+12)
  track.append(on)
  # Instantiate a MIDI note off event, append it to the track
  off = midi.NoteOffEvent(tick=2000, pitch=i+12)
  track.append(off)
  # Add the end of track event, append it to the track
  eot = midi.EndOfTrackEvent(tick=1)
  track.append(eot)
  # Print out the pattern
  # Save the pattern to disk
  midi.write_midifile("example.mid", pattern)

  # t = madmom.utils.midi.MIDITrack(track)
  m = md.MIDIFile.from_file("example.mid")

  # t.events = [madmom.utils.midi.Event(e)]
  # m = madmom.utils.midi.MIDIFile([t])
  notes = m.notes()

  #convert notes to input vectors between 0 and 1
  vector = []
  vector.append(notes[0][0]/7.29166667)
  vector.append((notes[0][1]-21)/24)
  vector.append(notes[0][2]/2.08333333)
  vector.append(notes[0][3]/100)
  vector.append(0)

  vector.append(notes[1][0]/7.29166667)
  vector.append((notes[1][1]-21)/24)
  vector.append(notes[1][2]/2.08333333)
  vector.append(notes[1][3]/100)
  vector.append(0)

  arr[x] = vector
  # print(notes)

f = h5py.File('midi_data.hdf5','w')  
dset = f.create_dataset("init", data=arr)

# This is how we will get the resulting notes from the netork into a midi file
# m = md.MIDIFile.from_notes(notes)
# m.write("example.mid")
