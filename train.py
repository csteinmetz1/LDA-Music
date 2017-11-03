import numpy as np
import matplotlib.pyplot as plt
import sunau
import os

import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import analysis # provides specilized methods for loading files and preprocessing

# generates "corpus" - list of bag of frequencies for audio excerpts
def generateCorpus():
    
    corpus = []
    rootdir = "genres/"

    for genre in os.listdir(rootdir):
        print genre
        if not genre.endswith(".DS_Store"):
            for filename in os.listdir(rootdir + genre):
                if filename.endswith(".au"):
                    print rootdir + genre + "/" + filename
                    data, fs = analysis.loadau(rootdir + genre + "/" + filename)
                    frames = analysis.frameSignal(data, fs)
                    bof = analysis.generateBagOfFrequencies(frames, fs)
                    corpus.append(bof)
        return corpus

def generateDictionary(sample_rate, fft_size):
    
    dictionary = {}
    fid = 0  # frequency id
    
    freqs = np.arange(0, float(sample_rate)/2.0, float(sample_rate)/float((fft_size)), dtype=float)

    for freq in freqs:
        dictionary[fid] = str(int(np.ceil(freq)))
        fid += 1 

    print dictionary
    return dictionary

def trainModel(dictionary, corpus, num_topics):

    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(corpus, num_topics=num_topics, id2word = dictionary, passes=1)

    return ldamodel

# perform the anaylsis here
dictionary = generateDictionary(22050, 2048)
corpus = generateCorpus()
ldamodel = trainModel(dictionary, corpus, 50)

ldamodel.save("model/lda.model")
