import numpy as np
import matplotlib.pyplot as plt
import sunau
import os

import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

ldamodel = gensim.models.LdaModel.load("model/lda.model")
print(ldamodel.print_topics(num_topics=50, num_words=5))