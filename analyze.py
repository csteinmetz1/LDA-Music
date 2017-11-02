import numpy as np
import matplotlib.pyplot as plt
import sunau

import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def loadAU(filename):
    signal_fp = sunau.open(filename, 'r')
    sample_rate = signal_fp.getframerate()
    num_frames = signal_fp.getnframes()

    signal_data = np.fromstring(signal_fp.readframes(num_frames), dtype=np.dtype('>h')) # convert data to array
    
    # normalization (don't want this for LDA model)
    #signal_data = signal_data.astype(float)
    #signal_data /= 32766.0

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
def generate_bof(frames, sample_rate, fft_size=2048, pow_fft=True):
    
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

# generates "corpus" - list of bag of frequencies for audio excerpts
def generate_corpus(bofs):
    pass

def generate_dictionary(sample_rate, fft_size):
    
    dictionary = {}
    fid = 0  # frequency id
    
    freqs = np.arange(0, float(sample_rate)/2.0, float(sample_rate)/float((fft_size)), dtype=float)

    for freq in freqs:
        dictionary[fid] = str(int(np.ceil(freq)))
        fid += 1 

    print dictionary
    return dictionary

def train_model(dictionary, corpus, num_topics):

    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(corpus, num_topics=num_topics, id2word = dictionary, passes=16)

    return ldamodel

rock01 = "genres/rock/rock.00005.au" # 30 sec clip from dataset
data, Fs = loadAU(rock01)

frames = frameSignal(data, Fs)
bof = generate_bof(frames, Fs)
dictionary = generate_dictionary(Fs, 2048)

corpus = []
corpus.append(bof)

train_model(dictionary, corpus, 50)


#plt.plot(bof[:][0], bof[:][1])
#plt.yscale('log')
#plt.xscale('log')
#plt.show()