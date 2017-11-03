import numpy as np
import matplotlib.pyplot as plt
import sunau
import os


def loadau(filename):
    """
    Load .au audio file.
    
    Args:
        filename: path and filename of desire file from the current directory

    Returns:
       singal_data: Numpy array of the audio data in int16 format
       sample_rate: Sample rate of the audio as an int 
    """
    signal_fp = sunau.open(filename, 'r')
    sample_rate = signal_fp.getframerate()
    num_frames = signal_fp.getnframes()

    signal_data = np.fromstring(signal_fp.readframes(num_frames), dtype=np.dtype('>h')) # convert data to array
    
    # normalization (don't want this for LDA model)
    #signal_data = signal_data.astype(float)
    #signal_data /= 32766.0

    # we should normalize each audio signal to 0dB to ensure louder
    # recordings do no have frequencies weighted more greatly

    # Also, we many want to apply loudness curves to the
    # frequency data in order to 

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

# generates bag of frequencies for one audio excerpt
def generateBagOfFrequencies(signal, sample_rate, fft_size=2048, pow_fft=True):
    
    frames = frameSignal(signal, sample_rate)

    mag_frames = np.absolute(np.fft.rfft(frames, fft_size))
    pow_frames = ((1.0 / fft_size) * ((mag_frames) ** 2))
    
    if pow_fft:
        freqVec = np.sum(pow_frames, axis=0)/len(pow_frames)
        freqVec = freqVec[0:len(freqVec)-1] # need to figure out why the length is too long
    else:
        freqVec = np.sum(mag_frames, axis=0)/len(mag_frames)

    bof = [] # bag of frequencies
    fid = 0  # frequency id

    # stuff bag of frequencies 
    for power in np.nditer(freqVec):
        bof.append((fid, int(power)))
        fid += 1

    return bof