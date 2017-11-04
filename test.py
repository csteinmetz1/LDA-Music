import numpy as np
import matplotlib.pyplot as plt
import sunau
import os

import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import analysis # provides specilized methods for loading files and preprocessing

# first load a file not used in training
data, fs = analysis.loadau("genres/rock/rock.00080.au")

# generate seed bag of frequencies vector
bof_seed = analysis.generateBagOfFrequencies(data, fs)

library = [] # to store list of all bof vectors for music library
rootdir = "genres/" # directory of dataset

# load all of the potential recommendations
for genre in os.listdir(rootdir):
    if not genre.endswith(".DS_Store"): # ignore these annoying MacOS X files
        print genre
        for filename in os.listdir(rootdir + genre):
            if filename.endswith(".au"): 
                print int(filename[filename.find(".")+1:filename.rfind(".")]) 
                data, fs = analysis.loadau(rootdir + genre + "/" + filename)
                bof = analysis.generateBagOfFrequencies(data, fs)
                library.append(bof)

# load the model from training
ldamodel = gensim.models.LdaModel.load("model/lda.model")

# we will find the topic distrobution for one song via the model
print ldamodel[bof]