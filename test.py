import numpy as np
import matplotlib.pyplot as plt
import sunau
import os

import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import analysis # provides specilized methods for loading files and preprocessing


# first load the file
data, fs = analysis.loadau("genres/rock/rock.00000.au")

# generate the bag of frequencies vector for the file
bof = analysis.generateBagOfFrequencies(data, fs)

# load the model from training
ldamodel = gensim.models.LdaModel.load("model/lda.model")

# we will find the topic distrobution for one song via the model
print ldamodel[bof]