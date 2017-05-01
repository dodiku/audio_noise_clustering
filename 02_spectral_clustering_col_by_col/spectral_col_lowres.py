import time
start_time = time.time()
recent_time = time.time()
file = open('02_spectral_clustering_col_by_col/timestamps.txt', 'w')

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
# spectragram = librosa.stft(y, hop_length=512, n_fft=512)
# spectragram = librosa.stft(y, hop_length=1024, n_fft=128)
# spectragram = librosa.stft(y, hop_length=64)
# spectragram = librosa.stft(y, n_fft=1024, hop_length=4096) # very small dataset

spectragram_final = librosa.stft(y, n_fft=4096, hop_length=128) # full
spectragram = librosa.stft(y, n_fft=1364, hop_length=3600) # low

rows, columns = spectragram.shape

print ('\n🎹  running audio analysis with data of', spectragram.shape, '\n')
print ('🎹  original spectrogram size is:', spectragram_final.shape, '\n')

plt.figure(1).set_size_inches(12,8)
plt.figure(1).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(librosa.amplitude_to_db(spectragram), cmap='rainbow')
plot_file = '02_spectral_clustering_col_by_col/pcolormesh.png'
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)
print ('🥝  pcolormesh plot is done.\n')


'''--------------------
performing spectral clusteting on the spectagram data
--------------------'''
# spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors", n_jobs=-1, assign_labels='discretize')
# spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors", n_jobs=-1, assign_labels='kmeans')
spectral = SpectralClustering(n_clusters=2)

# spectral_fit_predict = spectral.fit_predict(spectragram)
# spectral_fit_predict_reversed = spectral_fit_predict[::-1]


plt.figure(2).set_size_inches(12,8)
plt.figure(2).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)

all_labels = np.zeros((rows,columns))

for col in range (0, columns):
    spectral_fit_predict = spectral.fit_predict(spectragram[:,col].reshape(-1,1))
    # spectral_fit_predict_reversed = spectral_fit_predict[::-1]
    print ('🐙  clustering column', col, 'is done.')
    all_labels[:,col] = spectral_fit_predict
    x = np.full((rows, 1), col)
    y = np.linspace(0,rows,rows)
    plt.scatter(x,y, c=spectral_fit_predict, s=40, cmap="rainbow", alpha=0.2)

print ('🐙  clustering job is done.\n')
addTimestamp('clustering')

plot_file = '02_spectral_clustering_col_by_col/spectral_cluster.png'
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)
print ('🥝  clustering plot is done.\n')

# creating an array of lables in the size of the original recording
# all_labels = spectral_fit_predict
print ('~~~~~~',spectragram.shape, all_labels.shape,spectragram_final.shape)
all_labels = np.repeat(all_labels,int(spectragram_final.shape[0]/spectragram.shape[0]), axis=0)
all_labels = np.repeat(all_labels,int(spectragram_final.shape[1]/spectragram.shape[1]), axis=1)
print ('~~~~~~',spectragram.shape, all_labels.shape,spectragram_final.shape)
# print (spectral_fit_predict)
print (all_labels)

# all_labels = np.repeat(all_labels,round(final_spectragram.shape[0]/columns), axis=1)
plt.figure(3).set_size_inches(12,8)
plt.figure(3).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(all_labels, cmap="rainbow")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig('02_spectral_clustering_col_by_col/spectral_cluster_hres.png', dpi=300)
print ('🥝  clustering hi-res plot is done.\n')
recent_time = time.time()

'''--------------------
generating result01: all noise = 0
--------------------'''
spectragram2 = np.copy(spectragram_final)
# spectragram_db = librosa.amplitude_to_db(spectragram2)
spectragram3 = np.copy(spectragram_final)
rows2, columns2 = spectragram2.shape

for r in range(0, rows2):
    for c in range(0,columns2):
        if all_labels[r,c] == 0:
            # spectragram_db[r,c] = 0;
            spectragram2[r,c] = 0;
        else:
            spectragram3[r,c] = 0;
    # if spectral_fit_predict_reversed[r] == 0:
    #     for c in range(0,columns2):
    #         spectragram2[r,c] = 0

directory = '02_spectral_clustering_col_by_col/result01/'
output_file = directory + 'output.wav'
noise_file = directory + 'noise.wav'
plot_file = directory + 'spectral.png'

if not os.path.exists(directory):
    os.makedirs(directory)

# output = stft.ispectrogram(spectragram2)
# wavfile.write(output_file, fs, output)
output = librosa.core.istft(spectragram2,win_length=4096, hop_length=128)
librosa.output.write_wav(output_file,output,sr)
output = librosa.core.istft(spectragram3,win_length=4096, hop_length=128)
librosa.output.write_wav(noise_file,output,sr)
addTimestamp('result01')

plt.figure(4).set_size_inches(12,8)
plt.figure(4).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(librosa.amplitude_to_db(spectragram2), cmap="rainbow")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)


print ('🥝  result01 is done.\n')

'''--------------------
# generating result02: remove only possitive value
# --------------------'''
spectragram2 = np.copy(spectragram_final)
# spectragram_db = librosa.amplitude_to_db(spectragram2)
rows2, columns2 = spectragram2.shape

for r in range(0, rows2):
    for c in range(0,columns2):
        if all_labels[r,c] == 0:
            if spectragram2[r,c] > 0:
                # spectragram_db[r,c] = 0;
                spectragram2[r,c] = 0;

directory = '02_spectral_clustering_col_by_col/result02/'
output_file = directory + 'output.wav'
plot_file = directory + 'spectral.png'

if not os.path.exists(directory):
    os.makedirs(directory)

# output = stft.ispectrogram(spectragram2)
# wavfile.write(output_file, fs, output)
output = librosa.core.istft(spectragram2,win_length=4096, hop_length=128)
librosa.output.write_wav(output_file,output,sr)
addTimestamp('result02')

plt.figure(5).set_size_inches(12,8)
plt.figure(5).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(librosa.amplitude_to_db(spectragram2), cmap="rainbow")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)
print ('🥝  result02 is done.\n')


'''--------------------
generating result03: reduce possitive values
--------------------'''
spectragram2 = np.copy(spectragram_final)
rows2, columns2 = spectragram2.shape

for r in range(0, rows2):
    for c in range(0,columns2):
        if all_labels[r,c] == 0:
            if spectragram2[r,c] > 0:
                # spectragram_db[r,c] = 0;
                spectragram2[r,c] = spectragram2[r,c] * 0.2;

directory = '02_spectral_clustering_col_by_col/result03/'
output_file = directory + 'output.wav'
plot_file = directory + 'spectral.png'

if not os.path.exists(directory):
    os.makedirs(directory)

# output = stft.ispectrogram(spectragram2)
# wavfile.write(output_file, fs, output)
output = librosa.core.istft(spectragram2,win_length=4096, hop_length=128)
librosa.output.write_wav(output_file,output,sr)
addTimestamp('result03')

plt.figure(6).set_size_inches(12,8)
plt.figure(6).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(librosa.amplitude_to_db(spectragram2), cmap="rainbow")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)
print ('🥝  result03 is done.\n')

'''--------------------
generating result04: reduce all
--------------------'''
spectragram2 = np.copy(spectragram_final)
rows2, columns2 = spectragram2.shape

for r in range(0, rows2):
    for c in range(0,columns2):
        if all_labels[r,c] == 0:
            spectragram2[r,c] = spectragram2[r,c] * 0.2;
            # spectragram2[r,c] = spectragram2[r,c] * 0.2;
            # if spectragram2[r,c] > 0:
                # spectragram_db[r,c] = 0;


directory = '02_spectral_clustering_col_by_col/result04/'
output_file = directory + 'output.wav'
plot_file = directory + 'spectral.png'

if not os.path.exists(directory):
    os.makedirs(directory)

# output = stft.ispectrogram(spectragram2)
# wavfile.write(output_file, fs, output)
output = librosa.core.istft(spectragram2,win_length=4096, hop_length=128)
librosa.output.write_wav(output_file,output,sr)
addTimestamp('result04')

plt.figure(7).set_size_inches(12,8)
plt.figure(7).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(librosa.amplitude_to_db(spectragram2), cmap="rainbow")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)
print ('🥝  result04 is done.\n')

'''--------------------
generating result05: reduce possitive more
--------------------'''
spectragram2 = np.copy(spectragram_final)
rows2, columns2 = spectragram2.shape

for r in range(0, rows2):
    for c in range(0,columns2):
        if all_labels[r,c] == 0:
            if spectragram2[r,c] > 0:
                spectragram2[r,c] = 0;
            else:
                spectragram2[r,c] = spectragram2[r,c] * 0.45;


directory = '02_spectral_clustering_col_by_col/result05/'
output_file = directory + 'output.wav'
plot_file = directory + 'spectral.png'

if not os.path.exists(directory):
    os.makedirs(directory)

# output = stft.ispectrogram(spectragram2)
# wavfile.write(output_file, fs, output)
output = librosa.core.istft(spectragram2,win_length=4096, hop_length=128)
librosa.output.write_wav(output_file,output,sr)
addTimestamp('result05')

plt.figure(8).set_size_inches(12,8)
plt.figure(8).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(librosa.amplitude_to_db(spectragram2), cmap="rainbow")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)
print ('🥝  result05 is done.\n')



# plt.show()

end_time = time.time()
print ('\n~~~\n🕑  script run time (seconds) =', end_time-start_time, '\n')
finalTimestamp('total')
file.close
print ('🍕  dandy!\n~~~\n')
