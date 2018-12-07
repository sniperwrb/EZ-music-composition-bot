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
#ch=np.array([[48,52,55],[43,47,50],[45,48,52],[40,43,47],
#             [41,45,48],[36,40,43],[38,41,45],[43,47,50]])+12
ch=np.array([[48,52,55],[47,50,55],[48,52,57],[47,52,55],
             [48,53,57],[48,52,55],[50,53,57],[47,50,55]])
amp1=112
amp0=80
rhy=80
for i in range(l):
    if (i%8==0):
        a=(i//8)%8
        if (i>0):
            note_list.append(midi.NoteOffEvent(tick=rhy*(i-p), channel=0, data=[ch[a-1,0], 0]))
            note_list.append(midi.NoteOffEvent(tick=0, channel=0, data=[ch[a-1,1], 0]))
            note_list.append(midi.NoteOffEvent(tick=0, channel=0, data=[ch[a-1,2], 0]))
        p=i
        note_list.append(midi.NoteOnEvent(tick=0, channel=0, data=[ch[a,0], amp0]))
        note_list.append(midi.NoteOnEvent(tick=0, channel=0, data=[ch[a,1], amp0]))
        note_list.append(midi.NoteOnEvent(tick=0, channel=0, data=[ch[a,2], amp0]))
    if (melody[i]>0):
        if (i>0):
            note_list.append(midi.NoteOffEvent(tick=rhy*(i-p), channel=0, data=[k, 0]))
        k=melody[i]
        p=i
        note_list.append(midi.NoteOnEvent(tick=0, channel=0, data=[k, amp1]))

note_list.append(midi.NoteOffEvent(tick=rhy*(l-p), channel=0, data=[k, 0]))
note_list.append(midi.NoteOffEvent(tick=0, channel=0, data=[ch[-1,0], 0]))
note_list.append(midi.NoteOffEvent(tick=0, channel=0, data=[ch[-1,1], 0]))
note_list.append(midi.NoteOffEvent(tick=0, channel=0, data=[ch[-1,2], 0]))
note_list.append(midi.NoteOnEvent(tick=0, channel=0, data=[ch[0,0], amp0]))
note_list.append(midi.NoteOnEvent(tick=0, channel=0, data=[ch[0,1], amp0]))
note_list.append(midi.NoteOnEvent(tick=0, channel=0, data=[ch[0,2], amp0]))
note_list.append(midi.NoteOnEvent(tick=0, channel=0, data=[72, amp1]))
note_list.append(midi.NoteOffEvent(tick=rhy*8, channel=0, data=[72, 0]))
note_list.append(midi.NoteOffEvent(tick=0, channel=0, data=[ch[0,0], 0]))
note_list.append(midi.NoteOffEvent(tick=0, channel=0, data=[ch[0,1], 0]))
note_list.append(midi.NoteOffEvent(tick=0, channel=0, data=[ch[0,2], 0]))
note_list.append(midi.EndOfTrackEvent(tick=0, data=[]))
MIDI = midi.Pattern(tracks=[tone_set, note_list])
midi.write_midifile(time.strftime("%m%d%H%M%S")+".midi", MIDI)