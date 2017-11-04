import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
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
        #print genre
        for filename in os.listdir(rootdir + genre):
            if filename.endswith(".au"): 
                #print int(filename[filename.find(".")+1:filename.rfind(".")]) 
                data, fs = analysis.loadau(rootdir + genre + "/" + filename)
                bof = analysis.generateBagOfFrequencies(data, fs)
                library.append(bof)

# load the model from training
model = gensim.models.LdaModel.load("model/lda.model")

# we will find the topic distrobution for one song via the model
seed_topic_distribution = model.inference([bof_seed])[0][0]

# now find the distance to each other excerpt in the library
distance_list = []
for bof_candidate in library:
    
    # id and topic distribution as output from the model
    candidate_topic_distribution = model.inference([bof_candidate])[0][0]

    # calculate the distance from the seed    
    distance_list.append(euclidean(seed_topic_distribution, candidate_topic_distribution))

distance_array = np.asarray(distance_list)

# find the closet vector
closet = np.argmin(distance_array)
print distance_array[closet]
print closet
