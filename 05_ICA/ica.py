'''
flow:
    [v] 1 - read all files from a foder
    [ ] 2 - run ica on any two file
    [ ] 3 - output 2 files with names in series
    [ ] 4 - write to a summery file
'''
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
import seaborn as sns
from sklearn.decomposition import FastICA
from scipy.io import wavfile
import sys
import os


print(sys.argv)



def get_all_files(dir='/Users/dodiku/code/noise_clustering/05_ICA/spectral_results'):
    files_array = os.listdir(dir)
    if files_array[0] == '.DS_Store':
        files_array.remove(files_array[0])
    print ('📄  found', len(files_array), 'files\n')
    print (files_array)
    return files_array



def run_ica(array):

    tolerance = 0.001
    algorithm = 'parallel'
    # algorithm = 'deflation'

    file = open('05_ICA/ica_results/combinations.txt', 'w')
    dir_num = 0

    for first in array:
        for second in array:

            # if first == second:
            #     continue

            # creating a new directory
            directory = '05_ICA/ica_results/' + str(dir_num)
            if not os.path.exists(directory):
                os.makedirs(directory)

            # writing string to combinations.txt
            string = str(dir_num) + ' : ' + first + ' + ' + second + '\n'
            file.write(string)

            first_path = '05_ICA/spectral_results/' + first
            second_path = '05_ICA/spectral_results/' + second
            fs_1, voice_1 = wavfile.read(first_path)
            fs_2, voice_2 = wavfile.read(second_path)
            m, = voice_1.shape
            voice_2 = voice_2[:m]
            print ('🕵️‍ ', first, voice_1.shape, second, voice_2.shape)
            if voice_1.shape[0] > voice_2.shape[0]:
                voice_1 = np.resize(voice_1,(voice_2.shape[0],))
                print ("🕵️‍ 💯  Fixed array sizes :)")
            if voice_1.shape[0] < voice_2.shape[0]:
                voice_2 = np.resize(voice_2,(voice_1.shape[0],))
                print ("🕵️ ‍💯  Fixed array sizes :)")

            # plotting time domain representation of signal
            plt.figure(1).set_size_inches(12,8)
            plt.figure(1).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)

            plt.subplot(2, 1, 1)
            title = "Time Domain Representation of " + first
            plt.title(title)
            plt.xlabel("Time")
            plt.ylabel("Signal")
            plt.plot(np.arange(m)/fs_1, voice_1)

            plt.subplot(2, 1, 2)
            title = "Time Domain Representation of " + second
            plt.title(title)
            plt.xlabel("Time")
            plt.ylabel("Signal")
            plt.plot(np.arange(m)/fs_2, voice_2)

            plot_file = directory + '/TDR.png'
            plt.savefig(plot_file, dpi=300)

            # mix data
            voice = np.c_[voice_1, voice_2]
            A = np.array([[1, 0.5], [0.5, 1]])
            X = np.dot(voice, A)

            # plotting time domain representation of mixed signal
            plt.figure(2).set_size_inches(12,8)
            plt.figure(2).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)

            plt.subplot(2, 1, 1)
            title = "Time Domain Representation of mixed " + first
            plt.title(title)
            plt.xlabel("Time")
            plt.ylabel("Signal")
            plt.plot(np.arange(m)/fs_1, X[:, 0])

            plt.subplot(2, 1, 2)
            title = "Time Domain Representation of mixed " + second
            plt.title(title)
            plt.xlabel("Time")
            plt.ylabel("Signal")
            plt.plot(np.arange(m)/fs_2, X[:, 1])

            plot_file = directory + '/TDR_mixed.png'
            plt.savefig(plot_file, dpi=300)


            # blind source separation using ICA
            ica = FastICA(n_components=2, tol=tolerance, algorithm=algorithm)
            print ("🕵️‍  Training the ICA decomposer .....")
            t_start = time.time()
            ica.fit(X)
            t_stop = time.time() - t_start
            print ("🕵️‍  Training Complete; took %f seconds" % (t_stop))
            # get the estimated sources
            S_ = ica.transform(X)
            # get the estimated mixing matrix
            A_ = ica.mixing_


            # plotting time domain representation of estimated signal
            plt.figure(3).set_size_inches(12,8)
            plt.figure(3).subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.6, hspace=0.8)

            plt.subplot(2, 1, 1)
            title = "Time Domain Representation of estimated " + first
            plt.title(title)
            plt.xlabel("Time")
            plt.ylabel("Signal")
            plt.plot(np.arange(m)/fs_1, S_[:, 0])

            plt.subplot(2, 1, 2)
            title = "Time Domain Representation of estimated " + second
            plt.title(title)
            plt.xlabel("Time")
            plt.ylabel("Signal")
            plt.plot(np.arange(m)/fs_2, S_[:, 1])

            plot_file = directory + '/TDR_estimate.png'
            plt.savefig(plot_file, dpi=300)

            # output files
            file_one = directory + '/01.wav'
            file_two = directory + '/02.wav'
            wavfile.write(file_one, fs_1, S_[:,0]*40)
            wavfile.write(file_two, fs_1, S_[:,1]*40)

            print ('🐣  combination', dir_num+1, 'is done.\n')

            dir_num = dir_num + 1

    file.close
    return




array = get_all_files()
run_ica(array)


print ('👻  wooohaaa!\n~~~\n')
