import IPython.display as ipd
import math
import statistics
import numpy as np
from scipy.stats import entropy
from math import log, e
import pandas as pd
import timeit
import pyaudio
import wave
import copy
import matplotlib.pyplot as plt
import pyaudio
import struct 
import numpy as np
import time
import copy
CHUNK = 1024 * 2 
FORMAT =pyaudio.paInt16
CHANNELS = 1
RATE= 44100
N= 40000 #number of random numbers
L = 8   #ccml size
cc = 0.05 #coupling constant
epoch = 20 #number of iterations
n=int(N/32)
#Take audio file as input.
p= pyaudio.PyAudio()
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)
data=stream.read(CHUNK*8)

data_int= np.array(struct.unpack(str(CHUNK*2) +'q',data),dtype=np.int8)[::2]
t_end = time.time() 
data_int = np.empty( shape=(0, 0) )
#Convert the audio file into 8 bit variables.
while len(data_int)<(N*L)+2000:
    data=stream.read(CHUNK*8)
    t= np.array(struct.unpack(str(CHUNK*2) +'q',data),dtype=np.int8)
    data_int= np.append(data_int,t) 
data_int= data_int +127 
#Discarding Transcient Values 
o=data_int[1000:]
r=copy.deepcopy(o)


#Chaotic Map
def chaotic_map(x):
    alpha =1.99999
    if x<0.5:
        return (alpha* x)
    return (alpha*(1-x))
# o final array of 256 bit random numbers
o=[]
z=[0 for i in range(L)]

x = [[0 for i in range(epoch)] for j in range(L)]
x[0][0] = 0.141592
x[1][0] = 0.653589
x[2][0] = 0.793238
x[3][0] = 0.462643
x[4][0]= 0.383279
x[5][0]= 0.502884
x[6][0] = 0.197169
x[7][0] = 0.399375
t=0
c=0
y=0
while len(o)<= N:
    t=0
    
    for i in range(L):
        x[i][t]= float(((0.071428571 * (r[c]%8)) +x[i][t]) * 0.666666667)
        c  = c + 1
    for t in range (epoch):
        for i in range(L):
            x[i][(t+1)%epoch] = (1-cc)*x[i][t] + (cc/2)* (x[(i+1)%L][t]+ x[(i-1)%L][t])                    
    for i in range(L):
        z[i]= int(x[i][4]*(10**8) )%(2**32)
        x[i][0]= x[i][4]
        
    for i in range(int(L/2)):
        k=(z[i+4])
        swapped_k= ( (k<<(16))   | k>>(16) )#swap the 32 least significant bits(LSB) with 32 most significant bits(MSB).
        z[i]= (z[i] ^ swapped_k)#doing xor with z[i+4] here.
        z[i]= (z[i])%(2**32)
        
    print(k,"=k ",z[0]," ",z[1]," ",z[2]," ",z[3]," ")
    #Concatenate the resultant 64 bit variables to get 256 bit number.
    newnum = ( ( (  z[0] << 64  | z[1] ) << 64 | z[2] ) << 64 | z[3] )
    #o.append(newnum)
    o.append(z[0])
    o.append(z[1])
    o.append(z[2])
    o.append(z[3])



