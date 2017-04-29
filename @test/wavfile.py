import time
start_time = time.time()

import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import librosa


y, sr = librosa.load('_samples/01_counting.m4a') # audio from librosa
print ('y: ', np.min(y), np.max(y))
print (y)

stft = librosa.stft(y, hop_length=64)
stft_raw = librosa.stft(y, hop_length=64)
print ('stft_raw:')
print (stft_raw)
print (stft_raw.shape, '\n')
stft = librosa.amplitude_to_db(stft)
print ('stft:')
print (stft)
print (stft.shape, '\n')


'''
import stft
import scipy.io.wavfile as wav

fs, audio = wav.read('input.wav')
specgram = stft.spectrogram(audio)
output = stft.ispectrogram(specgram)
wav.write('output.wav', fs, output)
'''

# wavfile.write('output.wav', fs, output)
import stft as stft_lib
fs, audio = wavfile.read('_samples/01_counting_org.wav')
specgram = stft_lib.spectrogram(audio)
print ('specgram:')
print (specgram)
print (specgram.shape, '\n')
specgram_db = librosa.amplitude_to_db(specgram)
print ('specgram_db:')
print (specgram_db)
print (specgram_db.shape, '\n')

output = stft_lib.ispectrogram(specgram)
wavfile.write('output.wav', fs, output)

plt.figure(1)
plt.pcolormesh(specgram_db)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
plt.title('Spectragram of the raw audio recording,\nwhich was used as the dataset for this research')
# plt.savefig('_plots/pcolormesh.png', dpi=300)

plt.figure(2)
rows, columns = specgram_db.shape
X = np.zeros((columns*rows,3))

i = 0
for r in range(0, rows):
    for c in range(0,columns):
        X[i,:]=(c,r,specgram_db[r,c])
        i = i + 1

# np.savetxt("scatter.csv", X, delimiter=",")

# print ()
# print ()
# print ()
print ('X: ', X.shape, '\n')
# print (X)

# , cmap="YlGnBu"
plt.scatter(X[:,0], X[:,1], c=X[:,2], s=1)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Samples')
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
plt.title('Scatter plot of the spectragram of the raw audio recording,\nwhich was used as the dataset for this research\nEach data point was used as a single sample')

specgram2 = np.copy(specgram)

print ('specgram2 (before):')
print (specgram2)
print (specgram2.shape, '\n')

rows2, columns2 = specgram2.shape

for r in range(200, rows2):
    for c in range(0,columns):
        specgram2[r,c] = 0

print ('specgram2 (after):')
print (specgram2)
print (specgram2.shape, '\n')

output2 = stft_lib.ispectrogram(specgram2)
wavfile.write('output2.wav', fs, output2)


plt.show()
end_time = time.time()
print ('\n~~~\n🕑  script run time (seconds)= ', end_time-start_time, '\n')
print ('🌳  yas!\n~~~\n')
