import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
import sunau
import os

import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import analysis # provides specilized methods for loading files and preprocessing

# first load a file not used in training
data, fs = analysis.load_au("genres/rock/rock.00080.au")

# generate seed bag of frequencies vector
bof_seed = analysis.generateBagOfFrequencies(data, fs)

library = [] # to store list of all bof vectors for music library
titles = {} # to store song titles 
title = 0 # index to title dict

rootdir = "genres/" # directory of dataset

# load all of the potential recommendations
for genre in os.listdir(rootdir):
    if not genre.endswith(".DS_Store"): # ignore these annoying MacOS X files
        #print genre
        for filename in os.listdir(rootdir + genre):
            if filename.endswith(".au"): 
                #print int(filename[filename.find(".")+1:filename.rfind(".")]) 
                data, fs = analysis.load_au(rootdir + genre + "/" + filename)
                bof = analysis.generateBagOfFrequencies(data, fs)
                library.append(bof)
                titles[title] = filename
                title += 1

#data, fs = analysis.loadau("genres/rock/rock.00000.au")
#bof = analysis.generateBagOfFrequencies(data, fs)
#library.append(bof)

# load the model from training
model = gensim.models.LdaModel.load("model/lda.model")

# we will find the topic distrobution for one song via the model
seed_topic_distribution = model.inference([bof_seed])[0][0]

# now find the distance to each other excerpt in the library
distance_list = []
for bof_candidate in library:
    
    # topic distribution as output from the model
    candidate_topic_distribution = model.inference([bof_candidate])[0][0]

    # calculate the distance from the seed    
    distance_list.append(euclidean(seed_topic_distribution, candidate_topic_distribution))

distance_array = np.asarray(distance_list)

# find the closet vector
closest = np.argmin(distance_array) # closet value will be the seed in the library
distance_array /= closest # normalize by the seed distance (there is variation in the inference)

# find the 10 closest songs
inds = np.argpartition(distance_array, 10)[:10]
for ind in inds:
    print "%d %d %s" %(ind, distance_array[ind], titles[ind])

# now use PCA to reduce the dimensionality of each distibution vector
pca = PCA(n_components=2)
pca.fit(bof_seed)
bof_seed_new = pca.transform(bof_seed)

print bof_seed_new
