import time
start_time = time.time()
recent_time = time.time()
file = open('03_spectral_clustering_spec_3D/timestamps.txt', 'w')

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
# spectragram = stft.spectrogram(audio, framelength=256, overlap=1, save_settings=True) # low resolution
# spectragram = librosa.stft(y, hop_length=512, n_fft=512)
# spectragram = librosa.stft(y, hop_length=1024, n_fft=128)
# spectragram = librosa.stft(y, hop_length=64)
# spectragram = librosa.stft(y, n_fft=1024, hop_length=4096) # very small dataset
# spectragram_final = librosa.stft(y)

spectragram_final = librosa.stft(y, n_fft=4096, hop_length=128) # full
spectragram = librosa.stft(y, n_fft=1364, hop_length=3600) # low

# print (spectragram)
print ('\n🎹  running audio analysis with data of', spectragram.shape, '\n')
print ('🎹  original spectrogram size is:', spectragram_final.shape, '\n')

plt.figure(1).set_size_inches(12,8)
plt.figure(1).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(librosa.amplitude_to_db(spectragram), cmap='rainbow')
plot_file = '03_spectral_clustering_spec_3D/pcolormesh.png'
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)
print ('🥝  pcolormesh plot is done.\n')

'''--------------------
generating 3D dataset
--------------------'''
X = scatterReady(librosa.amplitude_to_db(spectragram))
X_final = scatterReady(librosa.amplitude_to_db(spectragram_final))
# print (X)

'''--------------------
performing spectral clusteting on the spectagram data
--------------------'''
# spectral = SpectralClustering(n_clusters=2, n_jobs=-1, eigen_solver='arpack', affinity="nearest_neighbors", assign_labels='discretize')
# spectral = SpectralClustering(n_clusters=2, n_jobs=-1, eigen_solver='arpack', affinity="nearest_neighbors", assign_labels='kmeans')
spectral = SpectralClustering(n_clusters=2, n_jobs=-1)

spectral_fit_predict = spectral.fit_predict(X)
addTimestamp('clustering')
print (spectral_fit_predict)
print ('🐙  clustering job is done.\n')
# spectral_fit_predict_reversed = spectral_fit_predict[::-1]


plt.figure(2).set_size_inches(12,8)
plt.figure(2).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)

# for col in range (0, columns):
#     spectral_fit_predict = spectral.fit_predict(spectragram[:,col].reshape(-1,1))
#     # spectral_fit_predict_reversed = spectral_fit_predict[::-1]
#     all_labels[:,col] = spectral_fit_predict
#     x = np.full((rows, 1), col)
#     y = np.linspace(0,rows,rows)
#     plt.scatter(X[:,0],X[:,1], c=spectral_fit_predict, s=1, cmap="rainbow")
plt.scatter(X[:,0],X[:,1], c=spectral_fit_predict, s=12, alpha=0.6, cmap="rainbow")
plot_file = '03_spectral_clustering_spec_3D/spectral_cluster.png'
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)
print ('🥝  clustering plot is done.\n')

# creating an array of lables in the size of the original recording
# all_labels = spectral_fit_predict
print ('~~~~~~',X.shape, spectral_fit_predict.shape,X_final.shape)
# all_labels = np.repeat(spectral_fit_predict[::-1],int(X_final.shape[0]/X.shape[0]), axis=0)
all_labels = np.repeat(spectral_fit_predict,int(X_final.shape[0]/X.shape[0]), axis=0)
print ('~~~~~~',X.shape, spectral_fit_predict.shape,X_final.shape, all_labels.shape)
# for i in range(0,X_final.shape[0]-all_labels.shape[0]):
#     all_labels = np.append(all_labels,all_labels[all_labels.shape[0]-1])
print ('~~~~~~',X.shape, spectral_fit_predict.shape,X_final.shape, all_labels.shape)
print (spectral_fit_predict)
print (all_labels)

# all_labels = np.repeat(all_labels,round(final_spectragram.shape[0]/columns), axis=1)
plt.figure(3).set_size_inches(12,8)
plt.figure(3).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.scatter(X_final[:,0],X_final[:,1], c=all_labels, s=12, alpha=0.6, cmap="rainbow")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig('03_spectral_clustering_spec_3D/spectral_cluster_hres.png', dpi=300)
print ('🥝  clustering hi-res plot is done.\n')
recent_time = time.time()


'''--------------------
generating result01: all noise = 0
--------------------'''
spectragram2 = np.copy(spectragram_final)
spectragram3 = np.copy(spectragram_final)
spectragram2_db = librosa.amplitude_to_db(spectragram2)
rows2, columns2 = spectragram2.shape
# print (rows2)
# print (spectragram2_db)

for r in range(0, all_labels.shape[0]):
    if all_labels[r] != all_labels[0]:
        spectragram2[int(X_final[r,1]), int(X_final[r,0])] = 0
        # spectragram2_db[int(X_final[r,1]), int(X_final[r,0])] = 0
    else:
        spectragram3[int(X_final[r,1]), int(X_final[r,0])] = 0

directory = '03_spectral_clustering_spec_3D/result01/'
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

plt.figure(1).set_size_inches(12,8)
plt.figure(1).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(librosa.amplitude_to_db(spectragram2), cmap="rainbow")
# print (spectragram2_db)
# plt.pcolormesh(spectragram2_db, cmap="rainbow")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)

print ('🥝  result01 is done.\n')
recent_time = time.time()

'''--------------------
generating result02: remove only possitive value
--------------------'''
spectragram2 = np.copy(spectragram_final)
# spectragram3 = np.copy(spectragram_final)
# spectragram2_db = librosa.amplitude_to_db(spectragram2)
rows2, columns2 = spectragram2.shape
# print (rows2)
# print (spectragram2_db)

for r in range(0, all_labels.shape[0]):
    if all_labels[r] != all_labels[0]:
        if spectragram2_db[int(X_final[r,1]), int(X_final[r,0])] > 0:
            spectragram2[int(X_final[r,1]), int(X_final[r,0])] = 0
            # spectragram2_db[int(X_final[r,1]), int(X_final[r,0])] = 0
        # for c in range(0,columns2):
        #
        #         spectragram2[r,c] = 0
        #         spectragram2_db[r,c] = 0

directory = '03_spectral_clustering_spec_3D/result02/'
output_file = directory + 'output.wav'
plot_file = directory + 'spectral.png'

if not os.path.exists(directory):
    os.makedirs(directory)

# output = stft.ispectrogram(spectragram2)
# wavfile.write(output_file, fs, output)
output = librosa.core.istft(spectragram2,win_length=4096, hop_length=128)
librosa.output.write_wav(output_file,output,sr)
addTimestamp('result02')

plt.figure(1).set_size_inches(12,8)
plt.figure(1).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
plt.pcolormesh(librosa.amplitude_to_db(spectragram2), cmap="rainbow")
# plt.pcolormesh(spectragram2_db, cmap="rainbow")
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.savefig(plot_file, dpi=300)
print ('🥝  result02 is done.\n')
recent_time = time.time()

'''--------------------
generating result03: reduce possitive values
--------------------'''
spectragram2 = np.copy(spectragram_final)
rows2, columns2 = spectragram2.shape

for r in range(0, all_labels.shape[0]):
    if all_labels[r] != all_labels[0]:
        if spectragram2_db[int(X_final[r,1]), int(X_final[r,0])] > 0:
            spectragram2[int(X_final[r,1]), int(X_final[r,0])] = spectragram2[int(X_final[r,1]), int(X_final[r,0])] * 0.2

directory = '03_spectral_clustering_spec_3D/result03/'
output_file = directory + 'output.wav'
plot_file = directory + 'spectral.png'

if not os.path.exists(directory):
    os.makedirs(directory)

output = librosa.core.istft(spectragram2,win_length=4096, hop_length=128)
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
spectragram2 = np.copy(spectragram_final)
rows2, columns2 = spectragram2.shape

for r in range(0, all_labels.shape[0]):
    if all_labels[r] != all_labels[0]:
        spectragram2[int(X_final[r,1]), int(X_final[r,0])] = spectragram2[int(X_final[r,1]), int(X_final[r,0])] * 0.2

directory = '03_spectral_clustering_spec_3D/result04/'
output_file = directory + 'output.wav'
plot_file = directory + 'spectral.png'

if not os.path.exists(directory):
    os.makedirs(directory)

output = librosa.core.istft(spectragram2,win_length=4096, hop_length=128)
librosa.output.write_wav(output_file,output,sr)
addTimestamp('result04')

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
spectragram2 = np.copy(spectragram_final)
rows2, columns2 = spectragram2.shape


for r in range(0, all_labels.shape[0]):
    if all_labels[r] != all_labels[0]:
        if spectragram2_db[int(X_final[r,1]), int(X_final[r,0])] > 0:
            spectragram2[int(X_final[r,1]), int(X_final[r,0])] = 0
        else:
            spectragram2[int(X_final[r,1]), int(X_final[r,0])] = spectragram2[int(X_final[r,1]), int(X_final[r,0])]  * 0.45


directory = '03_spectral_clustering_spec_3D/result05/'
output_file = directory + 'output.wav'
plot_file = directory + 'spectral.png'

if not os.path.exists(directory):
    os.makedirs(directory)

output = librosa.core.istft(spectragram2,win_length=4096, hop_length=128)
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
