import time
start_time = time.time()

import numpy as np
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import librosa
import stft
import seaborn as sns
sns.despine()
from scipy.io import wavfile
import pandas as pd

'''--------------------
loading audio file
--------------------'''
samples = ['01_counting_org.wav','02_wind_and_cars_org.wav','03_truck_org.wav','04_voices_org.wav','05_ambeint_org.wav','06_office_org.wav']
sample_file = '_samples/' + samples[0]

fs, audio = wavfile.read(sample_file)


'''--------------------
performing short time fourier transform
--------------------'''
specgram = stft.spectrogram(audio)


'''--------------------
performing spectral clusteting on the spectagram data
--------------------'''
# spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors", n_jobs=-1, assign_labels='discretize')
spectral = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="nearest_neighbors", n_jobs=-1, assign_labels='kmeans')
# spectral = SpectralClustering(n_clusters=2, eigen_solver='lobpcg', affinity="nearest_neighbors", n_jobs=-1, assign_labels='discretize')
# spectral = SpectralClustering(n_clusters=2, eigen_solver='lobpcg', affinity="nearest_neighbors", n_jobs=-1, assign_labels='kmeans')
# spectral = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_jobs=-1, assign_labels='discretize')
# spectral = SpectralClustering(n_clusters=2, affinity="nearest_neighbors", n_jobs=-1, assign_labels='kmeans')
# spectral = SpectralClustering(n_clusters=2, eigen_solver='amg', affinity="nearest_neighbors")
spectral_fit = spectral.fit(specgram)
# spectral_fit_predict = spectral.fit(specgram)
spectral_fit_predict = spectral_fit.fit_predict(specgram)

specgram_db = librosa.amplitude_to_db(specgram)
rows, columns = specgram_db.shape
X = np.zeros((columns*rows,3))
i = 0
for r in range(0, rows):
    for c in range(0,columns):
        X[i,:]=(c,r,specgram_db[r,c])
        i = i + 1

print (spectral_fit.labels_.shape)
print (specgram.shape)
print (X.shape)
plt.figure(1).set_size_inches(6,4)
plt.figure(1).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)
# plt.scatter(X[:,0], X[:,1], c=spectral_fit.labels_, s=1)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
# plt.title('Scatter plot of the spectragram of the raw audio recording,\nwhich was used as the dataset for this research\nEach data point was used as a single sample')

# plt.scatter(x_train.iloc[:,0].values,x_train.iloc[:,1].values, s=10, c=spectral_fit_model ,cmap='rainbow', alpha=0.6)

# error_spectral = 0
# for i in range(0,ts):
#     if spectral_fit_predict[i] != y_test[i]:
#         error_spectral = error_spectral + 1
# error_spectral = error_spectral/ts*100
# if error_spectral > 50:
#     error_spectral = 100 - error_spectral
# plt.scatter(x_test.iloc[:,0].values,x_test.iloc[:,1].values, s=10, c=spectral_fit_predict ,cmap='rainbow', alpha=0.6)
# plt.title("Spectral Clustering: Prediction (Error: %s%%)" %(round(error_spectral,2)), fontsize=9)
# plt.xticks(fontsize=7)
# plt.yticks(fontsize=7)
# plt.savefig('plots/spectral.png', dpi=300)


# , cmap="YlGnBu"
# cmap='rainbow'



'''--------------------
performing spectral clusteting on reorganized data
--------------------'''



'''--------------------
writing output .wav file
--------------------'''
specgram2 = np.copy(specgram)

print ('specgram2 (before):')
print (specgram2)
print (specgram2.shape, '\n')

rows2, columns2 = specgram2.shape

for r in range(22, rows2):
    for c in range(0,columns):
        specgram2[r,c] = 0

print ('specgram2 (after):')
print (specgram2)
print (specgram2.shape, '\n')

output = stft.ispectrogram(specgram2)
wavfile.write('01_spectral_clustering/output.wav', fs, output)

print ('output:')
print (output)

plt.figure(2)
plt.pcolormesh(librosa.amplitude_to_db(specgram2))

plt.show()

end_time = time.time()
print ('\n~~~\n🕑  script run time (seconds) =', end_time-start_time, '\n')
print ('🍕  dandy!\n~~~\n')
