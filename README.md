# True-Random-Number-Generation
<b>Generation of random number sequence by harnessing randomness in ambient sound. </b></br>

# ABSTRACT
True random number are used in many fields such as : Cryptography, statistics, scientific research , gambling ,and even gaming.</br>
Random Number Generators can be divided into two main categories deppending on unpredictability:</br>
<br>
<b> 1.Pseudo-Random Number Generators (PRNGs)</b></br>
 </br>
 <b>2.True Random Number Generators (TRNGs)</b> </br>
# PSEUDO-RANDOM NUMBER GENERATORS (PRNGs)</br>
The PRNG's use a seed value to generate random like sequences mathemetically by following an alogrithm.
However these produce the same sequence for same seed value.Though PRNGs pass the statistical tests of randomness it fails to pass non deterministic tests.
Since true randomness cannot be generated using a computer we harvest entropy from external phenomena.
# TRUE RANDOM NUMBER GENERATORS (TRNGs)
TRNGs can generate true random numbers at a high rate while maintaining strong stastical quality.These Generators are more computationally expensive as they need to harvest entropy from unpredictable physical phenomena such  as 1.Radioactive decay 2.electrical noise or even thermal fluctuations that are non deterministic.Compared to PRNGs ,TRNGs are usually slower as they harvest randomness from phyical phenomena.</br></br>

# CHAOS THEORY
A chaotic system is a mathematical function that depits aperiodicity , sensitivity , ergodicity , diffusion and confusion characteristics that fulfil  the requirements of cryptographic algorithms.Therefore they have been used in the design of encryption algorithms.</br>
A popular chaotic map used in many designs is the chaotic tent map used in many designs which maps the interval
[0,1] onto itselt:                     
![](https://github.com/AdarshSingh7647/True-Random-Number-Generation/blob/main/images/Trngchaotictentmap.png)            
where &alpha; is the coontrol parameter and x is the xurrent state of chaotic map.The behavior of a chaotic map is determined by its control paramenter value.</br>
```python
#Chaotic Map
def chaotic_map(x):
    alpha =1.99999
    if x<0.5:
        return (alpha* x)
    return (alpha*(1-x))
```
In the case of the tent map when &alpha; increases from 0 to the chaotic map evoles from periodic to aperiodic.</br>
This can be seen from the lyapunovs exponent plot of tent map where positive exponent indicate chaotic behaviour. </br>
![](https://github.com/AdarshSingh7647/True-Random-Number-Generation/blob/main/images/TentMapLyapunavExponent.png)           
 </br>   
In this TRNG we use chaotic system known as Coupld Chaotic Map Lattice (CCML).It has longer period length, higher complexity and has multiple positive lyapunov exponents.        
A chatic system with multiple postitive lyapunov's exponents is known as hyperchaotic.           

The CCML equation is as follows:                       </br>
![](https://github.com/AdarshSingh7647/True-Random-Number-Generation/blob/main/images/CCMLequaton.png)  </br> 
where &epsilon; is the coupling constant(&epsilon;=0.5) , &alpha;=1.99999 , i={1,2,3,.,L} , L is the size of the system (L=8) and f(x) is a local chaotic map. 
Examples for L=8 and L=7 are:</br>
![](https://github.com/AdarshSingh7647/True-Random-Number-Generation/blob/main/images/CCML7.png)![](https://github.com/AdarshSingh7647/True-Random-Number-Generation/blob/main/images/CCML8.png)</br>  

# AUDIO SAMPLING
The source of randomness is ambient sound which is recorded using a microphone and digitized.
Recordings are obtained at a sampling rate of 44.1kHz.Each sample is then stored as 8-bit variables.
To eliminate the transient effect at the start of the recording, first 10000 samples are discarded.

To achieve maximum entropy , the randomness of the audio clip is amplified using hyperchaos.
The CCML's states, x<sup>i</sup><sub>t</sub> are represented by 64bit IEEE double-precision binary floating point(FP) format.</br> 

# ALGORITHM
From each 8-bit sample taken from audio input, 3 LSB(Least significant bits) is used to perturb one chaotic state,therefore a total of eight samples are used at one time.

The interval between each pertubation operation is selected based on the number of rounds required to diffuse the bits in one state to all other states. For L=8 pertubation occures once every flr(8/2)=4 iterations.</br>
The value of x<sup>i</sup><sub>t</sub> is modified based on equation:</br>
<b> {x<sup>i</sup><sub>t</sub>= ((0.071428571 x r<sup>y</sup>) + x<sup>i</sup><sub>t</sub>) x 0.666666667} </b>   
where r<sup<y</sup> is the value of 3-bit random number obtained from the yth audio sample.This equation is cosen such that the value of x<sup>i</sup><sub>t</sub> will be modified by r<sup>y</sup> but still remain within the interval [0,1] as required for iteration of chaotic tent map.
</br>
The states for the CCML are,x<sup>i</sup><sub>0</sub>,i ={0,7} are initialized with arbitrary FP values</br>
(0.141592, 0.653589, 0.793238, 0.462643, 0.383279, 0.502884, 0.197169, 0.399375).</br>
These values are preturbed before the CCML is iterated four times.This ensures that the pertubation of each chaoic state has been dispersed to all chaotic states.</br>
![](https://github.com/AdarshSingh7647/True-Random-Number-Generation/blob/main/images/algorithm.png)</br>
z is the numpy array that stores final eight FP values, these values of z<sup>i</sup> are used to produce four 64bit numbers based on the equations:</br>

![](https://github.com/AdarshSingh7647/True-Random-Number-Generation/blob/main/images/znumpyeqn.png)</br>

where ![](https://github.com/AdarshSingh7647/True-Random-Number-Generation/blob/main/images/xor.png) is a bitwise exclusive-or(XOR) operation and swap function swaps 32 MSB(most significant bits) with 32 LSB(least significant bits).</br>
The XOR operation contributes to reducing bias in the final output.
By bitwise concatenation of z<sup>i</sup> values 256-bit random number sequence O is obtained.</br>
# CODE
``` python
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
import matplotlib.pyplot as plt
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
```
</br>
To test the randomness of the sequence generated,the sequence is run on the DIEHARD Test Suite which consists of 15 stastical tests the results for the tests can be seen the table below.</br>

| Test Name                                    | P-value     | Result |
|----------------------------------------------|-------------|--------|
| Birthday Spacings Test                       | 0.056737234 | PASSED |
| Overlapping 5-permutation Test               | 0.20026502  | PASSED |
| Binary Rank Test for 32x32 Matrices          | 0.95128235  | PASSED |
| Binary Rank Test for 6x8 Matrices            | 0.77978737  | PASSED |
| Bitstream Test                               | 0.35673184  | PASSED |
| Overlapping Pairs Sparse Occupancy Test      | 0.67582236  | PASSED |
| Overlapping Quadruples Sparse Occupancy Test | 0.08140074  | PASSED |
| DNA Spacing Test                             | 0.46612436  | PASSED |
| Count-The-1s Test on a Stream of Bytes       | 0.38779403  | PASSED |
| Count -The-1s Test for Specific Bytes        | 0.06658232  | PASSED |
| Parking Lot Test                             | 0.17962474  | PASSED |
| Minimum Distance Test                        | 0.03064587  | PASSED |
| 3D Spheres Test                              | 0.97024207  | PASSED |
| Overlapping Sums Test                        | 0.001004982 | WEAK   |
| Runs Test -1                                 | 0.12722730  | PASSED |
| Runs Test -2                                 | 0.23342902  | PASSED |
| Crap's Test                                  | 0.64883881  | PASSED |   

 </br>
If the P-value is < 0.01 or > 0.99 the generator has failed the test.        
The DIEHARD results in the table only lists worst case results for tests with multiple P-values.</br>
<b> The generator has passes all the tests</b>

# RESULTS
 ![](https://github.com/AdarshSingh7647/True-Random-Number-Generation/blob/main/images/diehardResults/DHnames.png)  

![](https://github.com/AdarshSingh7647/True-Random-Number-Generation/blob/main/images/diehardResults/DH0-3.png) 

![](https://github.com/AdarshSingh7647/True-Random-Number-Generation/blob/main/images/diehardResults/DH4-7.png)   

![](https://github.com/AdarshSingh7647/True-Random-Number-Generation/blob/main/images/diehardResults/DH8-12.png)  

![](https://github.com/AdarshSingh7647/True-Random-Number-Generation/blob/main/images/diehardResults/DH14-19.png)  
