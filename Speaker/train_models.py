import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from Speaker.speaker_fearures import extract_features
import pickle
import os
os.environ['OMP_NUM_THREADS'] = '8'

source = './speaker-identification/development_set'

train_file = 'speaker-identification/development_set_enroll.txt'

dest = './speaker_models/'

file_paths = open(train_file, 'r')

features = np.asarray(())
count = 1

for path in file_paths:
    path = path.strip()
    # path = os.path.join(source, path)
    print(source + '/' + path)

    #load the audio file
    sr, audio = read(source + '/' + path)

    # get the 40 features
    vector = extract_features(audio, sr)

    if features.size == 0:
        features = vector
    else:
        features = np.vstack((features, vector))

    # taring data is one person per 5 files
    if count == 5:
        gmm = GaussianMixture(n_components=16, max_iter=200, covariance_type='diag', n_init=3)
        gmm.fit(features)

        # saving the trained model
        pickleFile = path.split('-')[0] + '.gmm'
        pickle.dump(gmm, open(dest + pickleFile, 'wb'))

        features = np.asarray(())
        count = 0
    count = count + 1
