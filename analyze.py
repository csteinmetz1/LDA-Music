import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import sunau

def loadAU(filename):
    signal_fp = sunau.open(filename, 'r')
    sample_rate = signal_fp.getframerate()
    num_frames = signal_fp.getnframes()

    signal_data = np.fromstring(signal_fp.readframes(num_frames), dtype=np.dtype('>h')) # convert data to array

    return signal_data, sample_rate

def plotSignal(signal, sample_rate):

    signal = signal[range(sample_rate, sample_rate + 512)] # grab 512 samples

    Ts = 1.0/sample_rate # sample spacing
    n = len(signal) # number of sample points
    t = np.arange(0,n,1) 

    Y = scipy.fftpack.fft(signal)/n
    Y = Y[range(n/2)]
    Yt = np.arange(n/2)
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t,signal)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(Yt,abs(Y),'r') 
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.xscale('log')
    plt.yscale('log')

    plt.show()

rock01 = "genres/rock/rock.00000.au" # 30 sec clip from dataset
data, Fs = loadAU(rock01)
plotSignal(data, Fs)