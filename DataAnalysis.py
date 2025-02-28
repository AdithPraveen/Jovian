import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import glob
import jax
from tqdm import tqdm as progress_bar
import os

files = glob.glob("Data/*")

def plotting(files,i1,if_avg=True):
    t,V = np.empty((1,)),np.empty((1,))
    for i in range(len(files)):  
        df = pd.read_csv(files[0],skiprows=1)
        cols = df.columns
        t1,V1 = df[cols[0]],df[cols[1]]
        t = np.append(t,t1)
        V = np.append(V,V1)
        
        
    n = len(V)
    N_channel = 4096
    N_channels = N_channel*2
    t,V = np.array(t[:(n//N_channels)*N_channels]),np.array(V[:(n//N_channels)*N_channels])

    V = V.reshape(len(V)//N_channels,N_channels)
    V = jnp.array(V)

    def single_fft(arr):
        n = len(arr)
        _ = jnp.fft.fft(arr)
        return _[:n//2]

    vfft = jax.vmap(single_fft)

    out = vfft(V)
    
    if if_avg:
        num_avg = 10 # Number which is averaged
        
        out = out[:(out.shape[0]//num_avg)*num_avg][:]
        
        out = out.reshape(-1,num_avg,out.shape[1]).mean(axis=1)
    
    folder1 = "output_files_14_DEC_avg" # Folder for saving the output files
    folder2 = "output_images_14_DEC_avg" # Folder for saving the output images
    
    for folder in [folder1,folder2]:
        if not os.path.exists(folder):
            os.mkdir(folder)
            print(f"Folder {folder} made by python.")
            

    np.save(f"{folder1}/{i1}_array.npy",np.array(out))

    time_dev = np.abs(np.diff(t)[1])*1e-3
    total_time = time_dev*n
    time,freq = np.linspace(0,total_time,out.shape[0]),jnp.fft.fftfreq(N_channels,time_dev)[:N_channels//2]
    
    np.save(f"{folder1}/{i1}_time.npy",time)
    np.save(f"{folder1}/{i1}_freq.npy",freq)

    plt.figure(figsize=(10,6))
    plt.pcolormesh(freq,time,jnp.log(jnp.abs(out)),shading="nearest")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Time [s]")
    plt.title(f"Frame = {i1}")
    plt.savefig(f"{folder2}/{i1}_plot.jpeg",dpi=300)
    plt.clf()

N = len(files)
files = files[:(N//20)*20]
files = np.array(files).reshape(N//20,20)

for f in progress_bar(range(files.shape[0])):
    plotting(files[f],f)

print("Successfully over")


