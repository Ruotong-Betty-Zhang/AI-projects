import os
import pickle
import time

from scipy.io.wavfile import read
import numpy as np
from Speaker.speaker_fearures import extract_features

# paths
source = './speaker-identification/development_set'

test_file = 'speaker-identification/development_set_enroll.txt'

modelPath = './speaker_models/'

file_paths = open(test_file, 'r')
# get every model
gmm_files = [os.path.join(modelPath, fname) for fname in os.listdir(modelPath) if fname.endswith('.gmm')]

models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]

speakers = [fname.split("/")[-1].split(".gmm")[0] for fname in gmm_files]

for path in file_paths:
    path = path.strip()

    sr, audio = read(source + '/' + path)
    vector = extract_features(audio, sr)
    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    print("\tdetected as - ", speakers[winner])
    time.sleep(1.0)
