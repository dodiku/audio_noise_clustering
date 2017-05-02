# Speech Noise Clustering
An experiment with a variety of clustering (and clustering-like) techniques to reduce noise on a speech audio recording.

### Notable results

###### Highlighted results could be found [on this page](https://dodiku.github.io/audio_noise_clustering/results/).

![audio recording spectrogram](/results/images/pcolormesh.png)

![scatter plot of audio recording as data samples](/results/images/scatter.png)

![clustering](/results/images/spectral_cluster.png)

![audio output](/results/images/spectral.png)

![ICA](/results/images/TDR.png)

### Keynotes
- [Machine Learning for Cities](https://dodiku.github.io/audio_noise_clustering/keynote/Final_Project_Dror_Ayalon.pdf)
- [NOC](https://dodiku.github.io/audio_noise_clustering/keynote/Final_Project_Dror_Ayalon_NOC.pdf)

### Further work
- [ ] Add kmeans clustering
- [ ] Add hierarchical clustering
- [ ] Try to implement Word2vec
- [ ] Find more combinations between clustering algorithms and ICA
- [ ] Try to chain ICA results (use output as an input for the next run)
- [ ] Clean the code (it looks like a mess..)

### Attribution
Open-source python3 packages:
- [Scikit-learn](http://scikit-learn.org)
- [LibROSA](http://librosa.github.io/librosa/index.html)
- [Matplotlib](http://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org)
- [Pandas](http://pandas.pydata.org/)
- [blind_source_separation.py GitHub Gist](https://gist.github.com/abinashpanda/11113098) by [abinashpanda](https://github.com/abinashpanda)
