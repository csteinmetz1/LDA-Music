import analysis
import numpy as np

data, fs = analysis.load_mp3("1000Hz-5sec.mp3")
bof = analysis.generateBagOfFrequencies(data, fs)

frame = data[2048:2*2048]
frame *= np.hamming(len(frame))

analysis.plotSignal(frame, fs)
