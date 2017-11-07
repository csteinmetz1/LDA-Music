# LDA-Music
Using LDA topic modeling to generate music suggestions using a model trained on audio frequency data.

# Dataset
Currently using the GTZAN Genre Collection of 1000 song samples, each 30 seconds long. 
The tracks are 16/22050 mono files in the .au format. 
([http://marsyasweb.appspot.com/download/data_sets/](http://marsyasweb.appspot.com/download/data_sets/))

This data set came out of work from the following publication.

"Musical genre classification of audio signals" 
by G. Tzanetakis and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.

# Dependencies
Written for Python 2.7.13
* SciPy ([https://www.scipy.org/](https://www.scipy.org/))
* NumPy ([http://www.numpy.org/](http://www.numpy.org/))
* Pydub ([http://pydub.com/](http://pydub.com/))
* FFmpeg ([https://www.ffmpeg.org/](https://www.ffmpeg.org/))
* Matplotlib ([https://matplotlib.org/](https://matplotlib.org/))
* Gensim ([https://radimrehurek.com/gensim/](https://radimrehurek.com/gensim/))