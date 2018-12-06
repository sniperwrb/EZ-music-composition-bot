# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 10:15:36 2018

@author: wangruobai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import os

with open('small_data.txt','r') as f:
    raw_data=f.readlines()
a=[]
k=0
chords=[]
notes=[]
bars=[]
bar=0
for i in range(len(raw_data)):
    s=raw_data[i][:-1].split(' ')
    if (len(s)>=10):
        bar=bar+1
        chord=float(s[0])
        if (chord>=0):
            chord = int((chord*12/7-0.9)%12)
        else:
            chord = int((-chord*12/7-0.9)%12)+12
        for j in range(2,10):
            if (s[j][-1]!='-'):
                note=int((float(s[j])*12/7-0.9)%12)
                notes=notes+[note]
                chords=chords+[chord]
                bars=bars+[bar]
    else:
        l=len(notes)
        if (l>0):
            b=np.zeros((l,37),dtype=np.int32)
            for j in range(l):
                b[j,notes[j]]=1
                b[j,chords[j]+12]=1
                b[j,-1]=bars[j]
            a=a+[b]            
            chords=[]
            notes=[]
            bars=[]
            bar=0

def trans(A, y=0, x=0):
    x=x%A[-1,-1]
    p=0
    while (A[p,-1]<=x):
        p=p+1
    B=np.concatenate((A[p:,:-1],A[:p,:-1]),axis=0)
    f=np.arange(36)
    f=(f//12)*12+(f%12-y)%12
    B=B[:,f]
    return B
    #return torch.tensor(B).type(torch.FloatTensor)

class LSTM1(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(LSTM1, self).__init__()
        self.n_in=n_in
        self.n_hid=n_hid
        self.n_out=n_out
        self.lstm=nn.LSTM(self.n_in, self.n_hid)
        self.out1=nn.Linear(self.n_hid, self.n_out)
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (torch.zeros(1, 1, self.n_hid),
        torch.zeros(1, 1, self.n_hid))
    
    def forward(self,inputs):
        hidout,self.hidden=self.lstm(inputs.view(len(inputs),1,-1),self.hidden)
        #out,self.hidden=self.lstm(inputs,self.hidden)
        onehot=self.out1(hidout.view(len(inputs),-1))
        notes=F.softmax(onehot,dim=1)
        return notes


model=LSTM1(36,24,12)
load_old=True
if (load_old):
    model.load_state_dict(torch.load("test0.sav"))
    model.eval()
else:
    loss_fun=nn.MSELoss()
    optimizer=Adam(model.parameters(),lr=0.005,betas=(0.7,0.95))
    for epoch in range(30000):
        i=np.random.randint(len(a))
        y_trans = np.random.randint(12)
        x_trans = np.random.randint(1000)
        X=trans(a[i],y_trans,x_trans)
        Y=torch.tensor(X).type(torch.FloatTensor)
        model.zero_grad()
        model.hidden = model.init_hidden()
        notes_in = Y[:-1,:]
        notes_true = Y[1:,:12]
        
        notes_out = model(notes_in)
        loss=loss_fun(notes_out, notes_true)
        loss.backward()
        optimizer.step()
        if (epoch%100==0):
            print('Step:',epoch,'Loss:',loss.item())
    torch.save(model.state_dict(), "test0.sav")
    
chorden=[5,3,4]
chorden=chorden/np.sum(chorden)
chordmap=np.zeros((24,12))
for i in range(12):
    chordmap[i,i]=chorden[0]
    chordmap[i+12,i]=chorden[0]
    chordmap[i,(i+7)%12]=chorden[2]
    chordmap[i+12,(i+7)%12]=chorden[2]
    chordmap[i,(i+4)%12]=chorden[1]
    chordmap[i+12,(i+3)%12]=chorden[1]
chord_filt=np.array([1,0,1,0,1,1,0,1,0,1,0,1])
r=0.5 # the rate of LSTM-based instead of chord-based

'''
rhythm = np.array([1,0,1,0,1,1,1,1,  1,0,1,0,1,0,1,0,
                   1,0,1,0,1,1,1,1,  1,0,1,0,1,0,1,0,
                   0,0,1,1,1,0,1,0,  1,0,0,0,1,0,1,1,
                   1,0,1,0,1,0,1,0,  1,0,0,1,1,0,0,0])
'''

# Rhythm
def density_A(E, rat=0.0):
    y=np.array([rat*3,-rat,rat,0,rat*2,-rat,rat,0])
    return y+(E-np.mean(y))

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

chord = np.array([1,5,-6,-3,4,1,-2,5],dtype=np.float32)
#chord = np.array([1,1,1,1,1,1,1,1],dtype=np.float32)
chord = ((np.abs(chord)*12/7-0.9)%12).astype(np.int32)+12*(chord<0)
chord = np.tile(chord,5)
last_input = np.zeros((1,36))
last_input[0,7]=1
last_input[0,19]=1

melody = np.zeros_like(rhythm)
melody[-1]=67
note_filt_raw=np.zeros(12)
m_copy_rate = np.array([0,0,0,0,0,0,0,0,
                        0.80, 0.75, 0.69, 0.63, 0.57, 0.49, 0.40, 0.28,
                        0,0,0,0,0,0,0,0,
                        0,0,0,0,0,0,0,0,
                        0.80, 0.75, 0.69, 0.63, 0.57, 0.49, 0.40, 0.28])
with torch.no_grad():
    for h in range(np.shape(rhythm)[0]):
        if (rhythm[h]==0):
            melody[h]=melody[h-1]
        else:
            next_input = torch.tensor(last_input).type(torch.FloatTensor)
            next_note = model(next_input)
            #note = np.argmax(next_note)-3
            p=next_note.numpy()[0]
            #p=np.log(p*10000+1)
            q=chordmap[chord[h//8],:]
            p=p*r+q*(1-r)
            if (m_copy_rate[h//8]>0):
                copied_note=melody[h-64]%12
                p[copied_note]+=m_copy_rate[h//8]
                mcent=(melody[h-1]+melody[h-64]+61.5+h*0.0375)//3
            else:
                mcent=(melody[h-1]+61+h*0.0375)//2
            note_filt = np.exp(-note_filt_raw**2/2)
            p=p*chord_filt*note_filt
            note = np.random.choice(12, 1, p=p/np.sum(p))
            #print(note)
            #print(next_note)
            mcent=mcent-mcent//12
            melody[h]=(note-mcent)%12+mcent
            if (melody[h]<=mcent+3):
                pr=np.random.rand()*2
                if (pr<np.exp(-(mcent-melody[h])**2/8)):
                    melody[h]=melody[h]+12
            elif (melody[h]>=mcent+9):
                pr=np.random.rand()*2
                if (pr<np.exp(-(mcent+12-melody[h])**2/8)):
                    melody[h]=melody[h]-12
            last_input=np.zeros((1,36))
            last_input[0,note]=1
            last_input[0,chord[h//8]]=1
            note_filt_raw=note_filt_raw*0.6
            note_filt_raw[note]=note_filt_raw[note]+1
melody=melody*rhythm
print(np.reshape(melody,(-1,8,8)))

np.savetxt('save_test.txt',melody,fmt='%2d')
st=os.system("C:/Users/wangruobai/AppData/Local/Continuum/anaconda2/python untitled_recon.py")