# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:56:08 2018

@author: wrb
"""

import numpy as np
import midi
import time

melody=np.loadtxt('save_test.txt',dtype=np.int32)

# Recon
tone_set=[midi.TimeSignatureEvent(tick=0, data=[4, 2, 24, 8]),
                             midi.EndOfTrackEvent(tick=0, data=[])]
note_list=[midi.ProgramChangeEvent(tick=0, channel=0, data=[1])]
l=np.shape(melody)[0]
k=0
p=0
for i in range(l):
    if (melody[i]>0):
        if (i>0):
            note_list.append(midi.NoteOffEvent(tick=100*(i-p), channel=0, data=[k, 0]))
        k=melody[i]
        p=i
        note_list.append(midi.NoteOnEvent(tick=0, channel=0, data=[k, 111]))

note_list.append(midi.NoteOffEvent(tick=100*(l-p), channel=0, data=[k, 0]))
note_list.append(midi.NoteOnEvent(tick=0, channel=0, data=[72, 111]))
note_list.append(midi.NoteOffEvent(tick=800, channel=0, data=[72, 0]))
note_list.append(midi.EndOfTrackEvent(tick=0, data=[]))
MIDI = midi.Pattern(tracks=[tone_set, note_list])
midi.write_midifile(time.strftime("%m%d%H%M%S")+".midi", MIDI)