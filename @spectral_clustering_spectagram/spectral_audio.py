import time
start_time = time.time()

import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import librosa
import seaborn as sns
sns.despine()
import clean
import pandas as pd

'''--------------------
audio
--------------------'''
# plt.figure(2).set_size_inches(8,8)
# plt.figure(2).subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=1)

plt.figure(2).subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.5)

def read_file(file_name):
    sample_file = file_name
    sample_directory = '/Users/dodiku/code/noise_reduction/00_samples/'
    sample_path = sample_directory + sample_file

    # generating audio time series and a sampling rate (int)
    y, sr = librosa.load(sample_path)

    return y, sr

samples = ['01_counting.m4a','02_wind_and_cars.m4a','03_truck.m4a','04_voices.m4a','05_ambeint.m4a','06_office.m4a']
y, sr = read_file(samples[0])
# stft = librosa.stft(y, hop_length=64) # regular dataset
# stft = librosa.stft(y, n_fft=1024, hop_length=4096) # very small dataset
stft = librosa.stft(y, n_fft=7168, hop_length=10000)
stft = librosa.amplitude_to_db(stft)
print ('\n🎹  running audio analysis with data of', stft.shape, '\n')

plt.subplot(2,2,1)
plt.pcolormesh(stft, cmap="YlGnBu")
plt.ylabel('Frequency [Hz]', fontsize=6)
plt.xlabel('Samples', fontsize=6)
plt.title("stft pcolormesh", fontsize=10)
plt.xticks(fontsize=4)
plt.yticks(fontsize=4)
# plt.colorbar()
print ('\n📊  plot (2,2,1) is ready!\n')


plt.subplot(2,2,2)
rows, columns = stft.shape
X = np.zeros((columns*rows,3))
i = 0
for r in range(0, rows):
    for c in range(0,columns):
        X[i,:]=(c,r,stft[r,c])
        i = i + 1

plt.scatter(X[:,0], X[:,1], c=X[:,2], s=1)
plt.ylabel('Frequency [Hz]',fontsize=6)
plt.xlabel('Samples',fontsize=6)
plt.title("stft scatter", fontsize=10)
plt.xticks(fontsize=4)
plt.yticks(fontsize=4)
print ('\n📊  plot (2,2,2) is ready!\n')
# plt.colorbar()


plt.subplot(2,2,3)
print ('\n👾  running SpectralClustering with', rows*columns, 'data samples.\n')

# spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors", n_jobs=-1, assign_labels='discretize')
spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors", n_jobs=-1, assign_labels='kmeans')
# spectral = SpectralClustering(n_clusters=2, eigen_solver='lobpcg', affinity="nearest_neighbors", n_jobs=-1, assign_labels='discretize')
# spectral = SpectralClustering(n_clusters=2, eigen_solver='lobpcg', affinity="nearest_neighbors", n_jobs=-1, assign_labels='kmeans')
# spectral = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_jobs=-1, assign_labels='discretize')
# spectral = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_jobs=-1, assign_labels='kmeans')

# spectral = SpectralClustering(n_clusters=2, eigen_solver='amg', affinity="nearest_neighbors")

spectral_fit = spectral.fit_predict(X)
plt.scatter(X[:,0], X[:,1], c=spectral_fit, s=1, cmap="YlGnBu")
plt.title("stft spectral clustering", fontsize=10)
plt.xticks(fontsize=4)
plt.yticks(fontsize=4)
plt.ylabel('Frequency [Hz]',fontsize=6)
plt.xlabel('Samples',fontsize=6)
print ('\n📊  plot (2,2,3) is ready!\n')


plt.subplot(2,2,4)
print ('\n👾  running SpectralClustering per column for', rows*columns, 'data samples.\n')

for col in range (0, columns):
    df = pd.DataFrame(X)
    x = df.loc[X[:,0] == col]
    spectral_fit = spectral.fit_predict(x)
    plt.scatter(x.loc[:,0], x.loc[:,1], c=spectral_fit, s=1, cmap="YlGnBu")

plt.title("stft spectral clustering per column", fontsize=10)
plt.xticks(fontsize=4)
plt.yticks(fontsize=4)
plt.ylabel('Frequency [Hz]',fontsize=6)
plt.xlabel('Samples',fontsize=6)
print ('\n📊  plot (2,2,4) is ready!\n')


plt.savefig('plots/spectral_audio.png', dpi=300)

end_time = time.time()
print ('\n~~~\n🕑  script run time (seconds) =', end_time-start_time, '\n')
print ('🎹  audio clustering is done!\n~~~\n')
# plt.show()
