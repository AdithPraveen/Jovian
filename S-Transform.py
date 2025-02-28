import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,fftfreq
import multiprocessing as mp
import csv
from stockwell import st

# PROCESSING DATA

print("Code is starting...")
# Data importing

f = np.fromfile('gqrx_20230411_075922_20728200_2048000_fc.raw','float32')

print("Data imported")

f = f[::2]+1j*f[1::2]

n_parts = 30

st = []

fac = len(f)//30
freq = np.arange(fac)+2048000
t = np.arange(fac*30)

for i in range(n_parts):
    st.append(st.st(np.array(f[fac*i:fac*(i+1)]),hi=fac-1))
    t1 = t[fac*i:fac*(i+1)]
    plt.pcolormesh(t1,f,np.abs(st[i]),shading="nearest")
    plt.xlabel("Time (in s)")
    plt.ylabel("Frequency (in Hz)")
    
    # plt.pcolormesh(np.abs(st[i]))
    
    plt.savefig(f"{i}_gqrx_20230411_075922_20728200_2048000_fc.jpeg",dpi=200)
    plt.show()
    
print("Code over")
    



