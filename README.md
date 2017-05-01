# Speech Noise Clustering
An experiment with a variety of clustering (and clustering-like) techniques to reduce noise on a speech audio recording.

![audio recording spectrogram](/results/images/pcolormesh.png)

![scatter plot of audio recording as data samples](/results/images/scatter.png)

![clustering](/results/images/spectral_cluster.png)

![audio output](/results/images/spectral.png)


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
