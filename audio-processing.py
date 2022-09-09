import math as math #to use ceil method
import numpy as np #importing numpy to easily manipulate arrays
import matplotlib.pyplot as plt #importing matplotlib to plot the graph of the song
import sounddevice as sd #importing sounddevice to play the song
from scipy.fftpack import fft #to convert time domain signals to frequency domain signals 
# This the continuous time interval through which the song will be played
# [0.00000000e+00 ,2.44160495e-04 ,4.88320990e-04 ... 2.99951168e+00 ,2.99975584e+00, 3.00000000e+00]
t = np.linspace(0, 3 , 12*1024) 
# 3rd octave frequencies
C3 = 130.81
D3 = 146.83
E3 = 164.81
F3 = 174.61
G3 = 196
A3 = 220
B3 = 246.93
# 4th octave frequencies
C4 = 261.63
D4 = 293.66
E4 = 329.63
F4 = 349.23
G4 = 392
A4 = 440
B4 = 493.88

# The 2D array below is my song where each subarray contains a note 
# from the 3rd octave at index 0 and a not from the 4th octave at index 1 
song = np.array([[B3,B4],[B3,B4],[A3,A4],[A3,A4],[A3,A4],[G3,G4],[A3,A4]])

# The 2 1D arrays below together form the intervals which we will loop on 
# where the start of each interval is in the 1D array called start 
# and the end of each interval is correspondingly in the 1D array called end
start = np.array([0,0.73,1.3,1.53,1.9,2.3,2.64])
end = np.array([0.7,1.25,1.5,1.8,2.2,2.6,3])

# This is the counter which we will use as an index to loop on 
# the start and end arrays
j = 0 
# The 1D array below called x takes the shape of array t and is filled 
# with zeros and this the final array where we will add the final notes 
# and the values in this array will be plotted on the y-axis
x = np.zeros(np.shape(t))

# The for loop below is used to loop on the 2D array called song where 
# each sub array is called note and note[0] is a note from the 3rd octave
# and note[1] is a note from the 4th octave
for note in song:
    # The array below called start_of_interval it multiplies 
    # (sin(2𝜋*note[0]*𝑡)+sin(2𝜋*note[1]*𝑡))*𝑢(𝑡−start[j]) by the start of an interval at index j of the start array
    start_of_interval = np.reshape((np.sin(2*np.pi*note[0]*t) + np.sin(2*np.pi*note[1]*t))*[t>=start[j]],np.shape(t))
    
    # The array below called end_of_interval it multiplies 
    # (sin(2𝜋*note[0]*𝑡)+sin(2𝜋*note[1]*𝑡))*𝑢(𝑡−end[j]) by the end of an interval at the same index j of the end array
    end_of_interval = np.reshape((np.sin(2*np.pi*note[0]*t) + np.sin(2*np.pi*note[1]*t))*[t>=end[j]],np.shape(t))
   
    # Then we subtract the 2 arrays to get the time (interval) through 
    # which the note was played and add it to array x so the equation is
    # (sin(2𝜋*note[0]*𝑡)+sin(2𝜋*note[1]*𝑡))*(𝑢(𝑡−start[j])-𝑢(𝑡−end[j]))
    x += start_of_interval-end_of_interval
    
    # increment j by 1
    j+=1

# It plots the time domain graph of time (x-axis) 
#against x (y-axis) at row 1 col 1
plt.subplot(3,2,1)
plt.plot(t,x)

# It plays the song for 3 seconds and displays its sound
#sd.play(x,3*1024)

# number of samples is equal to the song_duration(3) x 1024
N = 3*1024 

#frequency axis range from 0 to 512 and its size is int(N/2)
f=np.linspace(0,512,int(N/2))

#this is the conversion of the time domain signal x 
#to frequency domain signal x_f
x_f_data= fft(x) 
x_f= 2/N * np.abs(x_f_data[0:np.int(N/2)])

#It plots the frequency domain graph of frequency f (x-axis) 
#against x_f (y-axis) at row 1 col 2
plt.subplot(3,2,2)
plt.plot(f,x_f)

#generate the noise signal by selecting the two random frequencies f1 and f2
f1,f2 = np.random.randint(0,512,2)

# 𝑛(𝑡)=sin(2*𝑓1*𝜋*𝑡)+sin(2𝑓2*𝜋*𝑡)
n = np.sin(2*f1*np.pi*t)+np.sin(2*f2*np.pi*t)

# add the noise signal n to time domain signal x
xn = n+x

#It plots the time domain graph of time t (x-axis) 
#against xn (y-axis) at row 2 col 1
plt.subplot(3,2,3)
plt.plot(t,xn)

#this is the conversion of the contaminated time domain signal xn
#to frequency domain signal xn_f
xn_f_data= fft(xn) 
xn_f= 2/N * np.abs(xn_f_data[0:np.int(N/2)])

#It plots the frequency domain graph of frequency f (x-axis) 
#against xn_f (y-axis) at row 2 col 2
plt.subplot(3,2,4)
plt.plot(f,xn_f)

# stores the maximum element in x_f
max_in_x_f = -1

# This loop gets the maximum note in x_f
for note in x_f:
    if note>max_in_x_f:
        max_in_x_f = note

# max_j stores the index of a peak in xn_f
max_j = -1

#index incremented to keep track of the location of elements in xn_f
j = 0

# max_k stores the previous peak of index max_j in xn_f
max_k = -1

#This loop gets the maximum 2 peaks in xn_f greater than  the max in x_f 
for note in xn_f:
    if note>math.ceil(max_in_x_f):
        max_k = max_j
        max_j = j
    j+=1

#stores one peak of xn_f at max_j index and round it up to next integer
#and subtracts by 1
f1_temp = math.ceil(f[max_j])-1

#stores second peak of xn_f at max_k index and round it up to next integer
#and subtracts by 1
f2_temp = math.ceil(f[max_k])-1


# n2(t) = sin(2*𝑓1_temp*𝜋*𝑡)+sin(2*𝑓2_temp*𝜋*𝑡)
n2 = np.sin(2*f1_temp*np.pi*t)+np.sin(2*f2_temp*np.pi*t)

#𝑥_𝑓𝑖𝑙𝑡𝑒𝑟𝑒𝑑(𝑡)=𝑥𝑛(𝑡)−[sin(2*𝑓1_temp*𝜋*𝑡)+sin(2*𝑓2_temp*𝜋*𝑡)]
x_filtered = xn-n2

#It plots the time domain graph of time t (x-axis) 
#against x_filtered (y-axis) at row 3 col 1
plt.subplot(3,2,5)
plt.plot(t,x_filtered)

#this is the conversion of the time domain signal x_filtered
#to frequency domain signal x_filtered_f
x_filtered_data= fft(x_filtered) 
x_filtered_f= 2/N * np.abs(x_filtered_data[0:np.int(N/2)])

#It plots the frequency domain graph of frequency f (x-axis) 
#against x_filtered_f (y-axis) at row 3 col 2
plt.subplot(3,2,6)
plt.plot(f,x_filtered_f)

#It plays the sound of x_filtered for 3 seconds
sd.play(x_filtered, 3*1024) 
