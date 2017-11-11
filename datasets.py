import os
import analysis

class Dataset:
    def __init__(self, datadir, corpusdir="corpus/", filename="AudioCorpus.txt"):
        self.datadir = datadir
        self.corpusdir = corpusdir
        self.filename = filename
        self.size = 0
        self.generateCorpusDocuments()

    def generateCorpusDocuments(self):
        """
        Iterates through the dataset and creates the bag of frequencies
        for each audio file and then saves the vectors to a text file.
        This prevents the need to perform the framing, windowing, and 
        FFT anaylsis when attempting to stream the corpus.

        """
        if not os.path.exists(self.corpusdir):
            print "Directory does not exsist. Creating " + self.corpusdir
            os.mkdir(self.corpusdir)

        self.size = 1

        fp = open(self.corpusdir + self.filename, "w") # open the file to store the vectors
        for audioFile, fs in self:
            bof = analysis.audio2bof(audioFile, fs)
            for freq in bof:
                fp.write(str(freq[1]) + " ")
            fp.write("\n")
            self.size += 1
        fp.close()

    def OpenAudioCorpus(self):
        self.AudioCorpus = AudioCorpus(corpusdir=self.corpusdir, filename=self.filename, size=self.size, fraction=self.fraction)
        return self.AudioCorpus


class AudioCorpus:
    """ This class implements a corpus object from the dataset.

    In order to limit the memory footprint of the training process
    this class enables the bag of frequency vectors to be streamed 
    into the LDA model serially.

    """
    def __init__(self, corpusdir, filename, size, fraction):
        self.corpusdir = corpusdir
        self.filename = filename
        self.fraction = fraction
        self.size = size
    def __iter__(self):
        for audioFileVec in open(self.corpusdir + "/" + self.filename):
            bof = []
            fid = 0
            for freq in audioFileVec.split():
                bof.append((fid, int(freq)))
                fid += 1
            yield bof

class GTZAN_Dataset(Dataset):
    """ This class implements a dataset object from the GTZAN dataset.

    This allows other methods to traverse the directory structure
    of the dataset on disk and see only the desireable audio files.

    """
    def __iter__(self):
        for genre in os.listdir(self.datadir):
            if not genre.endswith(".DS_Store"): # ignore these annoying MacOS X files
                for filename in os.listdir(self.datadir + genre):
                    if filename.endswith(".au"): # and int(filename[filename.find(".")+1:filename.rfind(".")]) < (self.fraction * 100): 
                        print int(filename[filename.find(".")+1:filename.rfind(".")]) 
                        data, fs = analysis.load_au(self.datadir + genre + "/" + filename)
                        yield data, fs

class FMA_SmallDataset:
    def __init__(self, fraction=1):
        self.rootdir = "fma_small/"
        self.fraction = fraction
    def __iter__(self):
        self.filesLoaded = 0 # reset this value everytime the object is iterated over
        for subdir in os.listdir(self.rootdir):
            if not subdir.endswith(".DS_Store"): # ignore these annoying MacOS X files
                for filename in os.listdir(self.rootdir + genre):
                    if filename.endswith(".mp3") and (self.filesLoaded / self.totalFiles) <= (self.fraction * 100.0): 
                        data, fs = analysis.load_mp3(self.rootdir + subdir + "/" + filename)
                        yield data, fs           