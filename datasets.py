import os
import analysis

class GTZAN_Dataset:
    """ This class implements a dataset object from the GTZAN dataset.

    This allows other methods to traverse the directory structure
    of the dataset on disk and see only the desireable audio files.

    """
    def __init__(self, fraction=1):
        self.rootdir = "genres/"
        self.fraction = fraction
    def __iter__(self):
        for genre in os.listdir(self.rootdir):
            if not genre.endswith(".DS_Store"): # ignore these annoying MacOS X files
                for filename in os.listdir(self.rootdir + genre):
                    if filename.endswith(".au") and int(filename[filename.find(".")+1:filename.rfind(".")]) < (self.fraction * 100): 
                        print int(filename[filename.find(".")+1:filename.rfind(".")]) 
                        data, fs = analysis.load_au(self.rootdir + genre + "/" + filename)
                        yield data, fs

class GTZAN_AudioCorpus:
    """ This class implements a corpus object from the GTZAN dataset.

    In order to limit the memory footprint of the training process
    this class enables the bag of frequency vectors to be streamed 
    into the LDA model serially.

    """
    def __init__(self):
        self.corpusDirectory = "corpus/GTZAN_AudioCorpus.txt"
    def __iter__(self):
        for audioFileVec in open(self.corpusDirectory):
            bof = []
            fid = 0
            for freq in audioFileVec.split():
                bof.append((fid, int(freq)))
                fid += 1
            yield bof

class FMA_SmallDataset:
    def __init__(self, fraction=1):
        self.rootdir = "genres/"
        self.fraction = fraction

class FMA_AudioCorpus:
    def __iter__(self):
        for song in library: # placeholder code
            singal, fs = analysis.load_mp3(song)
            bof = analysis.audio2bof(signal, fs)
            yield bof