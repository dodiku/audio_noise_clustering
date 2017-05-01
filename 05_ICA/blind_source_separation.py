#!/usr/bin/env python

from __future__ import division

"""
https://gist.github.com/abinashpanda/11113098
Independent component analysis (ICA) is used to estimate
sources given noisy measurements. Imagine 2 persons speaking
simultaneously and 2 microphones recording the mixed signals.
ICA is used to recover the sources ie. what is said by each person.
"""

import time
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.io import wavfile

# loading wav files
fs_1, voice_1 = wavfile.read(
                    # "/Users/dodiku/Desktop/_samples/03_truck_org.wav")
                    "/Users/dodiku/code/noise_clustering/03_spectral_clustering_spec_3D/result02/output.wav")
                    # "/Users/dodiku/code/noise_clustering/01_spectral_clustering_spec/arpack_kmeans/result01/output.wav")
                    # "/Users/dodiku/code/noise_clustering/01_spectral_clustering_spec/none_kmeans/result01/output.wav")
                    # "/Users/dodiku/code/noise_clustering/01_spectral_clustering_spec/arpack_kmeans/result05/output.wav")
                    # "/Users/dodiku/Desktop/output02.wav")

fs_2, voice_2 = wavfile.read(
                    # "/Users/dodiku/code/noise_clustering/01_spectral_clustering_spec/none_kmeans/result01/output.wav")
                    # "/Users/dodiku/code/noise_clustering/01_spectral_clustering_spec/arpack_kmeans/result03/output.wav")
                    # "/Users/dodiku/code/noise_clustering/01_spectral_clustering_spec/none_kmeans/result03/output.wav")
                    "/Users/dodiku/code/noise_clustering/03_spectral_clustering_spec_3D/result01/noise.wav")
                    # "/Users/dodiku/Desktop/output01.wav")
# reshaping the files to have same size
print ('~~~voice_1:')
print (voice_1, '\n')
print ('~~~voice_2:')
print (voice_2, '\n')

m, = voice_1.shape
voice_2 = voice_2[:m]

# plotting time domain representation of signal
figure_1 = plt.figure("Original Signal")
plt.subplot(2, 1, 1)
plt.title("Time Domain Representation of voice_1")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.plot(np.arange(m)/fs_1, voice_1)
plt.subplot(2, 1, 2)
plt.title("Time Domain Representation of voice_2")
plt.plot(np.arange(m)/fs_2, voice_2)
plt.xlabel("Time")
plt.ylabel("Signal")

# mix data
voice = np.c_[voice_1, voice_2]
print ('~~~voice_1:')
print (voice_1, '\n')
print ('~~~voice_2:')
print (voice_2, '\n')
print ('~~~voice:')
print (voice, '\n')
A = np.array([[1, 0.5], [0.5, 1]])
X = np.dot(voice, A)
# X = voice
print ('~~~X:')
print (X, '\n')
# plotting time domain representation of mixed signal
figure_2 = plt.figure("Mixed Signal")
plt.subplot(2, 1, 1)
plt.title("Time Domain Representation of mixed voice_1")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.plot(np.arange(m)/fs_1, X[:, 0])
plt.subplot(2, 1, 2)
plt.title("Time Domain Representation of mixed voice_2")
plt.plot(np.arange(m)/fs_2, X[:, 1])
plt.xlabel("Time")
plt.ylabel("Signal")

# blind source separation using ICA
# ica = FastICA(n_components=2, tol=0.01, algorithm='deflation')
ica = FastICA(n_components=2, tol=0.001, algorithm='parallel')
print ("Training the ICA decomposer .....")
t_start = time.time()
ica.fit(X)
t_stop = time.time() - t_start
print ("Training Complete; took %f seconds" % (t_stop))
# get the estimated sources
S_ = ica.transform(X)
print ('~~~S_:')
print (S_, '\n')
# get the estimated mixing matrix
A_ = ica.mixing_
# assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# plotting time domain representation of estimated signal
figure_3 = plt.figure("Estimated Signal")
plt.subplot(2, 1, 1)
plt.title("Time Domain Representation of estimated voice_1")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.plot(np.arange(m)/fs_1, S_[:, 0])
plt.subplot(2, 1, 2)
plt.title("Time Domain Representation of estimated voice_2")
plt.plot(np.arange(m)/fs_2, S_[:, 1])
plt.xlabel("Time")
plt.ylabel("Signal")


wavfile.write('/Users/dodiku/Desktop/output01.wav', fs_1, S_[:,0]*40)
wavfile.write('/Users/dodiku/Desktop/output02.wav', fs_1, S_[:,1]*40)


# plt.show()
