import subprocess
import os

for mp3_file in os.listdir(os.getcwd()):
    if mp3_file.endswith("mp3"):
        filename = os.path.splitext(mp3_file)[0]
        in_file  = filename + ".mp3"
        out_file = "wav/" + filename + ".wav"

subprocess.call("ffmpeg -i " + in_file + " -ac 1 -ar 22050 " + out_file, shell=True)