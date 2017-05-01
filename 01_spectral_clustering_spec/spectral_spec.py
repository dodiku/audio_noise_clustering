import time
start_time = time.time()
recent_time = time.time()
file = open('01_spectral_clustering_spec/timestamps.txt', 'w')

import os
import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import librosa
import stft
import seaborn as sns
sns.despine()
from scipy.io import wavfile
import pandas as pd

def scatterReady(spec):
    rows, columns = spec.shape
    X = np.zeros((columns*rows,3))
    i = 0
    for r in range(0, rows):
        for c in range(0,columns):
            X[i,:]=(c,r,spec[r,c])
            i = i + 1
    return X

# stage == string
def addTimestamp(stage, recent_time=recent_time):
    string = stage + ': ' + str(time.time()-recent_time) + '\n'
    recent_time = time.time()
    file.write(string)

def finalTimestamp(stage):
    string = stage + ': ' + str(time.time()-start_time) + '\n'
    file.write(string)


'''--------------------
loading audio file
--------------------'''
samples = ['01_counting_org.wav','02_wind_and_cars_org.wav','03_truck_org.wav','04_voices_org.wav','05_ambeint_org.wav','06_office_org.wav']
sample_file = '_samples/' + samples[1]

# fs, audio = wavfile.read(sample_file)
y, sr = librosa.load(sample_file)

'''--------------------
performing short time fourier transform
--------------------'''
# spectragram = stft.spectrogram(audio)
spectragram = librosa.stft(y)


'''--------------------
performing spectral clusteting on the spectagram data
--------------------'''
# spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors", n_jobs=-1, assign_labels='discretize')
spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors", n_jobs=-1, assign_labels='kmeans')
# spectral = SpectralClustering(n_clusters=2)

spectral_fit_predict = spectral.fit_predict(spectragram)
spectral_fit_predict_reversed = spectral_fit_predict[::-1]
addTimestamp('clustering')
print ('🐙  clustering job is done.\n')

'''--------------------
generating result01: all noise = 0
--------------------'''
spectragram2 = np.copy(spectragram)
spectragram3 = np.copy(spectragram)
rows2, columns2 = spectragram2.shape

for r in range(0, rows2):
    if spectral_fit_predict_reversed[r] == 0:
        for c in range(0,columns2):
            spectragram2[r,c] = 0
    else:
        for c in range(0,columns2):
            spectragram3[r,c] = 0

directory = '01_spectral_clustering_spec/result01/'
output_file = directory + 'output.wav'
noise_file = directory + 'noise.wav'
plot_file = directory + 'spectral.png'

if not os.path.exists(directory):
    os.makedirs(directory)

# output = stft.ispectrogram(spectragram2)
# wavfile.write(output_file, fs, output)
output = librosa.core.istft(spectragram2)
librosa.output.write_wav(output_file,output,sr)
output = librosa.core.istft(spectragram3)
librosa.output.write_wav(noise_file,output,sr)
addTimestamp('result01')

plt.figure(1).set_size_inches(12,8)
plt.figure(1).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(librosa.amplitude_to_db(spectragram2), cmap="rainbow")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)

plt.figure(2).set_size_inches(12,8)
plt.figure(2).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
X = scatterReady(librosa.amplitude_to_db(spectragram2))
# spectral_fit_predict_X = spectral.fit_predict(X)

bd = librosa.amplitude_to_db(spectragram2)

for i in range (0, spectral_fit_predict_reversed.shape[0]):
    if spectral_fit_predict_reversed[i] == 0:
        y = np.full((columns2, 1), i)
        x = np.linspace(0,columns2,columns2)
        plt.scatter(x, y, c='purple', s=3, alpha=0.6)
    else:
        y = np.full((columns2, 1), i)
        x = np.linspace(0,columns2,columns2)
        plt.scatter(x, y, c='navy', s=3, alpha=0.6)

plot_file = '01_spectral_clustering_spec/spectral_cluster.png'
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)

print ('🥝  result01 is done.\n')
recent_time = time.time()

'''--------------------
generating result02: remove only possitive value
--------------------'''
spectragram2 = np.copy(spectragram)
rows2, columns2 = spectragram2.shape

for r in range(0, rows2):
    if spectral_fit_predict_reversed[r] == 0:
        for c in range(0,columns2):
            if spectragram2[r,c] > 0:
                spectragram2[r,c] = 0

directory = '01_spectral_clustering_spec/result02/'
output_file = directory + 'output.wav'
plot_file = directory + 'spectral.png'

if not os.path.exists(directory):
    os.makedirs(directory)

# output = stft.ispectrogram(spectragram2)
# wavfile.write(output_file, fs, output)
output = librosa.core.istft(spectragram2)
librosa.output.write_wav(output_file,output,sr)
addTimestamp('result02')

plt.figure(1).set_size_inches(12,8)
plt.figure(1).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(librosa.amplitude_to_db(spectragram2), cmap="rainbow")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)
print ('🥝  result02 is done.\n')
recent_time = time.time()

'''--------------------
generating result03: reduce possitive values
--------------------'''
spectragram2 = np.copy(spectragram)
rows2, columns2 = spectragram2.shape

for r in range(0, rows2):
    if spectral_fit_predict_reversed[r] == 0:
        for c in range(0,columns2):
            if spectragram2[r,c] > 0:
                spectragram2[r,c] = spectragram2[r,c] * 0.2

directory = '01_spectral_clustering_spec/result03/'
output_file = directory + 'output.wav'
plot_file = directory + 'spectral.png'

if not os.path.exists(directory):
    os.makedirs(directory)

# output = stft.ispectrogram(spectragram2)
# wavfile.write(output_file, fs, output)
output = librosa.core.istft(spectragram2)
librosa.output.write_wav(output_file,output,sr)
addTimestamp('result03')

plt.figure(1).set_size_inches(12,8)
plt.figure(1).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(librosa.amplitude_to_db(spectragram2), cmap="rainbow")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)
print ('🥝  result03 is done.\n')
recent_time = time.time()

'''--------------------
generating result04: reduce all
--------------------'''
spectragram2 = np.copy(spectragram)
rows2, columns2 = spectragram2.shape

for r in range(0, rows2):
    if spectral_fit_predict_reversed[r] == 0:
        for c in range(0,columns2):
            spectragram2[r,c] = spectragram2[r,c] * 0.2

directory = '01_spectral_clustering_spec/result04/'
output_file = directory + 'output.wav'
plot_file = directory + 'spectral.png'
addTimestamp('result04')

if not os.path.exists(directory):
    os.makedirs(directory)

# output = stft.ispectrogram(spectragram2)
# wavfile.write(output_file, fs, output)
output = librosa.core.istft(spectragram2)
librosa.output.write_wav(output_file,output,sr)

plt.figure(1).set_size_inches(12,8)
plt.figure(1).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(librosa.amplitude_to_db(spectragram2), cmap="rainbow")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)
print ('🥝  result04 is done.\n')
recent_time = time.time()

'''--------------------
generating result05: reduce possitive more
--------------------'''
spectragram2 = np.copy(spectragram)
rows2, columns2 = spectragram2.shape

for r in range(0, rows2):
    if spectral_fit_predict_reversed[r] == 0:
        for c in range(0,columns2):
            if spectragram2[r,c] > 0:
                spectragram2[r,c] = 0
            else:
                spectragram2[r,c] = spectragram2[r,c] * 0.45


directory = '01_spectral_clustering_spec/result05/'
output_file = directory + 'output.wav'
plot_file = directory + 'spectral.png'

if not os.path.exists(directory):
    os.makedirs(directory)

# output = stft.ispectrogram(spectragram2)
# wavfile.write(output_file, fs, output)
output = librosa.core.istft(spectragram2)
librosa.output.write_wav(output_file,output,sr)
addTimestamp('result05')

plt.figure(1).set_size_inches(12,8)
plt.figure(1).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(librosa.amplitude_to_db(spectragram2), cmap="rainbow")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)
print ('🥝  result05 is done.\n')
recent_time = time.time()


# plt.show()

end_time = time.time()
print ('\n~~~\n🕑  script run time (seconds) =', end_time-start_time, '\n')
finalTimestamp('total')
file.close
print ('🍕  dandy!\n~~~\n')
