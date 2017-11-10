import numpy as np
import matplotlib.pyplot as plt
import sunau
import os

import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import analysis # provides specilized methods for loading files and preprocessing

# generates "corpus" - list of bag of frequencies for audio excerpts
# we will split the data set and use 80% to train the model
# NOTE: This uses the structure for the GTZAN dataset
def generateCorpus():
    """ to be repalaced by streaming corpus classes """
    corpus = []
    rootdir = "genres/"
    for genre in os.listdir(rootdir):
        if not genre.endswith(".DS_Store"): # ignore these annoying MacOS X files
            print genre
            for filename in os.listdir(rootdir + genre):
                # use the first 80 excerpts from each genre
                if filename.endswith(".au") and int(filename[filename.find(".")+1:filename.rfind(".")]) < 80: 
                    print int(filename[filename.find(".")+1:filename.rfind(".")]) 
                    data, fs = analysis.load_au(rootdir + genre + "/" + filename)
                    bof = analysis.audio2bof(data, fs)
                    corpus.append(bof)
    return corpus

def generateDictionary(sample_rate, fft_size):
    dictionary = {}
    fid = 0  # frequency id
    freqs = np.arange(0, float(sample_rate)/2.0, float(sample_rate)/float((fft_size)), dtype=float)
    for freq in freqs:
        dictionary[fid] = str(int(np.ceil(freq)))
        fid += 1 
    return dictionary

def generateCorpusDocuments():
    """
    Iterates through the dataset and creates the bag of frequencies
    for each audio file and then saves the vectors to a text file.
    This prevents the need to perform the framing, windowing, and 
    FFT anaylsis when attempting to stream the corpus.

    """
    fp = open("corpus/GTZAN_AudioCorpus.txt", "w") # open the file to store the vectors
    dataset = GTZAN_Dataset(0.8) # instantiate iterable dataset object

    for audioFile, fs in dataset:
        bof = analysis.audio2bof(audioFile, fs)
        for freq in bof:
            fp.write(str(freq[1]) + " ")
        fp.write("\n")

    fp.close()


def trainModel(dictionary, corpus, num_topics):
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel = Lda(corpus, num_topics=num_topics, id2word=dictionary, passes=16, minimum_probability=0.0)
    return ldamodel

### begin training ###
if not os.path.exists("corpus/GTZAN_AudioCorpus.txt"):
    os.makedirs("corpus/")
    generateCorpusDocuments()

dictionary = generateDictionary(22050, 2048)
corpus = GTZAN_AudioCorpus()
ldamodel = trainModel(dictionary, corpus, 50)

if not os.path.exists("model/"):
    os.mkdir("corpus/")

ldamodel.save("model/lda.model")

