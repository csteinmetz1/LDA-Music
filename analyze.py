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

    Ts = 1.0/sample_rate # sample spacing
    n = len(signal) # number of sample points
    t = np.arange(0,n,1) 

    Y = np.absolute(np.fft.rfft(signal, n))
    P = ((1.0 / n) * ((Y) ** 2)) # power spectrum
    Y = Y[range(n/2)]
    P = P[range(n/2)]
    Yt = np.arange(n/2)
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(t,signal)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')
    ax[1].plot(Yt,abs(P),'r') 
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')
    plt.xscale('log')
    plt.yscale('log')

    plt.show()

def frameSignal(signal, sample_rate, frame_length=2048, frame_step=1024):
    
    signal_length = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) 

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros(pad_signal_length - signal_length)
    pad_signal = np.append(signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # apply window
    frames *= np.hamming(frame_length)

    return frames

def genFreqVec(frames, sample_rate, num_freq=2048, pow_fft=True):
    
    mag_frames = np.absolute(np.fft.rfft(frames, num_freq))
    pow_frames = ((1.0 / num_freq) * ((mag_frames) ** 2))

    if pow_fft:
        freqVec = np.sum(pow_frames, axis=0)
    else:
        freqVec = np.sum(mag_frames, axis=0)

    return freqVec


rock01 = "genres/rock/rock.00000.au" # 30 sec clip from dataset
data, Fs = loadAU(rock01)
frames = frameSignal(data, Fs)
freqVec = genFreqVec(frames, Fs)
freq = np.arange(1025)

plt.plot(freq, freqVec)
plt.yscale('log')
plt.xscale('log')
plt.show()