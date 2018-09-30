# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 12:56:08 2018

@author: wrb
"""

import numpy as np
import midi
import time

def density_G(E, rat=1.0):
    y=np.array([rat**3,1/rat,rat,1,rat**2,1/rat,rat,1])
    return y*(E/np.mean(y))

def density_A(E, rat=0.0):
    y=np.array([rat*3,-rat,rat,0,rat*2,-rat,rat,0])
    return y+(E-np.mean(y))

def chord_expand(c, w=[7,6,5,4,3,2,1]):
    l=np.shape(w)[0]
    if (l<7):
        if (l==0):
            w=[1,1,1,1,1,1,1]
        elif (l==1):
            w=[w[0],w[0],w[0],1,1,1,1]
        elif (l==2):
            w=[w[0],w[1],w[1],1,1,1,1]
        elif (l==3):
            w=[w[0],w[1],w[2],1,1,1,1]
        elif (l==4):
            w=[w[0],w[1],w[2],w[3],w[3],1,1]
        elif (l==5):
            w=[w[0],w[1],w[2],w[3],w[4],1,1]
        elif (l==6):
            w=[w[0],w[1],w[2],w[3],w[4],w[5],1]
    k = int((12.0/7)*np.abs(c)-13.0/14)
    if (c>=0):
        v=np.array([w[0],w[6], w[3],w[6], w[2], w[4],w[5], 
                    w[1],w[6], w[3],w[5], w[4]])
        v=np.concatenate((v[(12-k):],v[0:(12-k)]))
    else:
        v=np.array([w[0],w[5], w[4], w[1],w[6], w[3],w[6],
                    w[2], w[4],w[5], w[3],w[6]])
        v=np.concatenate((v[(12-k):],v[0:(12-k)]))
    return v
    
# Rhythm
r_copy_from = np.array([0, 0, 0,1, 0,1,2,3, 0,1,2,3,4,5,6,7, 14,15])
r_copy_rate = np.array([0.00,  0.20,  0.40, 0.28,  0.60, 0.52, 0.42, 0.30, 
               0.80, 0.75, 0.69, 0.63, 0.57, 0.49, 0.40, 0.28,  0.35, 0.25])
r_copy_from_short = r_copy_from[[0,1,2,3,4,5,6,7,14,15]]
r_copy_rate_short = r_copy_rate[[0,1,2,3,4,5,6,7,14,15]] 
# Verse
rhythm_v = []
v_density = density_A(0.55,0.2)
v_extra = 0
for bar in range(16+2*v_extra):
    for note in range(8):
        r=np.random.rand()
        if (bar==0):
            th=v_density[note]
        else:
            th=r_copy_rate[bar]*rhythm_v[r_copy_from[bar]*8+note]+\
                (1-r_copy_rate[bar])*v_density[note]
        if (r>th):
            rhythm_v.append(0)
        else:
            rhythm_v.append(1)
        
# Prechorus
rhythm_p = []
p_density = density_A(0.6,0.18)
p_extra = 0
for bar in range(8+2*p_extra):
    for note in range(8):
        r=np.random.rand()
        if (bar==0):
            th=p_density[note]
        else:
            th=r_copy_rate_short[bar]*rhythm_p[r_copy_from_short[bar]*8+note]+\
                (1-r_copy_rate_short[bar])*p_density[note]
        if (r>th):
            rhythm_p.append(0)
        else:
            rhythm_p.append(1)
            
# Chorus
rhythm_c = []
c_density = density_A(0.65,0.16)
c_extra = 0
have_coda = False
for bar in range(16+2*c_extra):
    for note in range(8):
        r=np.random.rand()
        if (bar==0):
            th=c_density[note]
        else:
            th=r_copy_rate[bar]*rhythm_c[r_copy_from[bar]*8+note]+\
                (1-r_copy_rate[bar])*c_density[note]
        if (have_coda)and(bar==15+2*c_extra):
            th=th*(1-note/8.0)
        if (r>th):
            rhythm_c.append(0)
        else:
            rhythm_c.append(1)
        
rhythm = np.concatenate((rhythm_v, rhythm_p, rhythm_c))

# Chord
# Placeholder
chord_v = [1,5,-6,-3,4,1,-2,5,1,5,-6,-3,4,1,-2,5]
chord_p = [1,5,-6,-3,4,1,-2,5]
chord_c = [1,5,-6,-3,4,1,-2,5,1,5,-6,-3,4,1,-2,5]
chord = np.concatenate((chord_v, chord_p, chord_c))

# Melody
m_copy_from = np.array([0, 0, 0,1, 0,1,2,3, 0,1,2,3,4,5,6,7, 14,15])
m_copy_rate = np.array([0,0,0,0,0,0,0,0, 
               0.80, 0.75, 0.69, 0.63, 0.57, 0.49, 0.40, 0.28,  0.35, 0.25])
m_copy_from_short = m_copy_from[[0,1,2,3,4,5,6,7,14,15]]
m_copy_rate_short = m_copy_rate[[0,1,2,3,4,5,6,7,14,15]]
Force_on_chord = False
pdist_sparse = [5,4,3,0,0,0,0]#[5.6,3.6,2.2,0.8,0.5,0.3,0.2]
fdist_sparse = [1,1,1,0,0,0,0]
last_note=12

# Verse
melody_v = []
v_mean = 13
v_var = 6**2
v_multiplier = np.arange(37)
v_multiplier = np.exp(-((v_multiplier-v_mean)**2)/(2.0*v_var))
for bar in range(16+2*v_extra):
    pdist=chord_expand(chord_v[bar],pdist_sparse)
    pdist=np.concatenate((pdist,pdist,pdist,pdist[0:1]))*v_multiplier
    if (Force_on_chord):
        fdist=chord_expand(chord_v[bar],fdist_sparse)
        fdist=np.concatenate((fdist,fdist,fdist,fdist[0:1]))*v_multiplier
    else:
        fdist=pdist
    for note in range(8):
        ndist=(fdist if note==0 else pdist)
        if (last_note>12):
            ndist[0:last_note-12]=0
        if (last_note<24):
            ndist[last_note+13:37]=0
        ndist=ndist/np.sum(ndist)
        if (bar>=8):
            copied_note=melody_v[m_copy_from[bar]*8+note]
            if (ndist[copied_note]>0):
                ndist=ndist*(1-m_copy_rate[bar])
                ndist[copied_note]+=m_copy_rate[bar]
        this_note=np.random.choice(37,p=ndist)
        if (rhythm_v[bar*8+note]>0):
            last_note=this_note
        melody_v.append(this_note)
        
# Prechorus
melody_p = []
p_mean = 18
p_var = 6**2
p_multiplier = np.arange(37)
p_multiplier = np.exp(-((p_multiplier-p_mean)**2)/(2.0*p_var))
for bar in range(8+2*p_extra):
    pdist=chord_expand(chord_p[bar],pdist_sparse)
    pdist=np.concatenate((pdist,pdist,pdist,pdist[0:1]))*p_multiplier
    if (Force_on_chord):
        fdist=chord_expand(chord_p[bar],fdist_sparse)
        fdist=np.concatenate((fdist,fdist,fdist,fdist[0:1]))*p_multiplier
    else:
        fdist=pdist
    for note in range(8):
        ndist=(fdist if note==0 else pdist)
        if (last_note>12):
            ndist[0:last_note-12]=0
        if (last_note<24):
            ndist[last_note+13:37]=0
        ndist=ndist/np.sum(ndist)
        if (bar>=8):
            copied_note=melody_p[m_copy_from[bar]*8+note]
            if (ndist[copied_note]>0):
                ndist=ndist*(1-m_copy_rate[bar])
                ndist[copied_note]+=m_copy_rate[bar]
        this_note=np.random.choice(37,p=ndist)
        if (rhythm_p[bar*8+note]>0):
            last_note=this_note
        melody_p.append(this_note)
            
# Chorus
melody_c = []
c_mean = 25
c_var = 6**2
c_multiplier = np.arange(37)
c_multiplier = np.exp(-((c_multiplier-c_mean)**2)/(2.0*c_var))
for bar in range(16+2*c_extra):
    pdist=chord_expand(chord_c[bar],pdist_sparse)
    pdist=np.concatenate((pdist,pdist,pdist,pdist[0:1]))*c_multiplier
    if (Force_on_chord):
        fdist=chord_expand(chord_c[bar],fdist_sparse)
        fdist=np.concatenate((fdist,fdist,fdist,fdist[0:1]))*c_multiplier
    else:
        fdist=pdist
    for note in range(8):
        ndist=(fdist if note==0 else pdist)
        if (last_note>12):
            ndist[0:last_note-12]=0
        if (last_note<24):
            ndist[last_note+13:37]=0
        ndist=ndist/np.sum(ndist)
        if (bar>=8):
            copied_note=melody_c[m_copy_from[bar]*8+note]
            if (ndist[copied_note]>0):
                ndist=ndist*(1-m_copy_rate[bar])
                ndist[copied_note]+=m_copy_rate[bar]
        this_note=np.random.choice(37,p=ndist)
        if (rhythm_c[bar*8+note]>0):
            last_note=this_note
        melody_c.append(this_note)
       
if (have_coda):
    p=bar*8+note
    while (rhythm_c[p]==0):
        p=p-1
    melody_c[p]=24
melody = np.concatenate((melody_v, melody_p, melody_c))
melody = (melody+48) * rhythm
print(np.reshape(melody,(-1,8,8)))

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