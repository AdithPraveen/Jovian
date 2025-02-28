import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftfreq
import multiprocessing as mp
import csv

# PROCESSING DATA

print("Code is starting...")
# Data importing

f = np.fromfile('gqrx_20230411_075922_20728200_2048000_fc.raw','float32')

print("Data imported")

f = f[::2]+1j*f[1::2]
print(f)

# Defining variables
N = len(f)
avg = 20 #Averaging number
f_channel = 2048 #Number of frequency channels
n_arr = N//(f_channel*avg) 
# This (n_arr) is the final number of arrays we will get in the 2D array after short time Fourier transform and averaging 
sample_rate = 2048000
central_frequency = 20728200

# Slicing the data for better data processing and plotting
f=f[:f_channel*avg*n_arr]

# Defining the hanning function
def u(n):
    return (1/2)*(1 - np.cos(2*np.pi*n/(f_channel-1)))

Han_t = np.array(range(f_channel))
Han = u(Han_t)
# print(Han)

'''
Here we are first defining a dictionary. The key will be the index of each array and the value will be
the FFT of that time series array. This is done to make averaging of these arrays easier. In FFT the
order hass been swapped as it gives posoitive frequency points before negative frequency points.
'''

whole_data={}
for i in range(avg*n_arr):
    # Applying Hanning window and slicing the data
    data = np.array(f[(i)*f_channel:(i+1)*f_channel])*Han 
    # print(data)
    data_length = len(data)
    # Performing Fourier Transform
    shft = fft(data)
    
    # Taking the absolute sqaure and rearranging the points
    spect = abs(np.array(list(shft[data_length//2:])+list(shft[:data_length//2])))**2
    whole_data[i]=spect

print("Short time Fourier Transform done")
print("The total number of arrays before averaging :",len(whole_data))

'''
In averaging we first define a list. We then take the corresponding number of arrays in order from the 
dictionary defined earlier, average them and then we will append it to the new list. In this way we will
get the 2D array. 
'''
avg_data=[]
for i in range(n_arr):
    temp_data=0
    # The below step is averaging. Note that every value in dictionary whole_data is a numpy array and 
    # so temp_data will turn into a numpy array
    for j in range(i*avg,avg*(i+1)):
        temp_data+=whole_data[j]
    
    # Normalizing it
    temp_data = list(temp_data/avg)
    
    avg_data.append(temp_data)
'''
NOTE:
At this points avg_data and every element of it is a list, that is it is a nested list.
'''

# Resolution and total time of data
time_resolution = (f_channel*avg)/sample_rate
frequency_resolution = sample_rate/f_channel
total_time = (sample_rate**-1)*len(f)
print("Time resolution is {} seconds".format(time_resolution))
print("Total time analysed is {} seconds".format(total_time))
print("Frequency resolution is {} Hz".format(frequency_resolution))

# PLOTTING DATA

'''
NOTE:
For creating a dynamic spectrum plot the dimensions of 2D array along each axis should be 1 less than the
number of elements in axis.

NOTE:
In IQ sampling we can distinguish negative and positive frequencies, doubling the bandwidth.

'''

# Taking the appropriate transpose of the data

temp_data = []
for i in range(f_channel-1):
    temp = []
    for j in range(len(avg_data)):
        temp.append(avg_data[j][i])
    temp_data.append(temp)

avg_data = 0
avg_data = temp_data
# Creating freqeuncy axis
temp_freq_axis  = fftfreq(f_channel,d=1/sample_rate)+central_frequency
freq_axis = np.array(list(temp_freq_axis[:f_channel//2])+list(temp_freq_axis[f_channel//2:]))

# Creating total time axis. This differs from truncated time axis as plot is seperated into 30 parts
total_time_axis = np.arange(0,total_time,time_resolution)
# Seperating avg_data into 30 parts and plotting each part seperately


    
# Saving data
filename = f"Total_data_10_April_Processed_avg_{avg}_fchannels_{f_channel}.csv"
fields = [i for i in range(f_channel)]

with open(filename, 'a') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(avg_data)

print("Data saving done")


parts = 30
_ = len(avg_data[1])//parts
for i in range(len(avg_data)):
    p = avg_data[i].copy()
    p = p[:_*parts]
    avg_data[i] = p

# Creating a new temperary list plt_data containing corresponding part of avg_data
# Each of this plot is for approximately 1 minute

print("Plotting starting...")
for i in range(parts):
    fig,ax = plt.subplot()
    plt_data = [avg_data[j][_*i:_*(i+1)-1] for j in range(f_channel-1)] # This is one less than required as for 2D plot this is necessary
    trunc_time_axis = total_time_axis[_*i:_*(i+1)]
    ax.pcolormesh(trunc_time_axis,freq_axis,plt_data,cmap="cividis")
    ax.set_title(f"10 April Part {i}")
    ax.set_xlabel("Time (in seconds)")
    ax.set_ylabel("Frequency (in Hz)")
    ax.colorbar()
    fig.savefig(f"10_April_Part_{i+1}_avg_{avg}_fchannels_{f_channel}_New_Analysis.jpeg",dpi=200)
    plt.show()

print("Plotting done")
print("All Completed Successfully")
