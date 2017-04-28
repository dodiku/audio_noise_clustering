import sys
import numpy as np
import matplotlib.pyplot as plt
import plot_graph as plg
import pandas as pd

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

#df = pd.read_csv('/Users/lgnonato/Meusdocs/Cursos/CUSP-GX-5006/Data/manhattan-dof.csv',index_col=False,delimiter=';')
#
#df_x = df.ix[:,3:6]
#df_y = df.ix[:,1]
#X = df_x.as_matrix()
#Y = df_y.as_matrix()

#####################
# Simulated data
#####################
m = 100
t = np.linspace(0,2*3.1415,m)
X = np.zeros((m,3))
X1 = np.zeros((m,3))
X[:,0] = 4*np.cos(t)
X[:,1] = 4*np.sin(t)
X[:,2] = 0
X1[:,0] = 2*np.cos(t)
X1[:,1] = 2*np.sin(t)
X1[:,2] = 1
X = np.vstack((X,X1))
pert = np.random.uniform(low=-0.5, high=0.5, size=(2*m,2))
X = X[:,0:2] + pert


#####################
# sklearn spectral clustering
#####################
Xspec = SpectralClustering(n_clusters = 2)
cl=Xspec.fit(X).labels_
#print(cl.shape)
plt.figure(1)
plt.scatter(X[:,0],X[:,1],c=cl)

#####################
# our spectral clustering
#####################

#####
# sklearn Knn
#####
neigh=NearestNeighbors(n_neighbors=6)
neigh.fit(X)
W = neigh.kneighbors_graph()


#####
# Our Knn
#####
n = X.shape[0]
W = np.zeros((n,n))
for i in range(1,n-1):
   for j in range(i+1,n):
       W[i,j] = np.exp(-(np.linalg.norm(X[i,0:2]-X[j,0:2],axis=None))/0.1)
       W[j,i]=W[i,j]


#####
# Eigen Vectors and Values of Laplacian
#####
D = np.diag(np.sum(W,axis=0))
L = D - W
S,U = np.linalg.eigh(L)

km = KMeans(n_clusters=2, random_state=0)
km.fit(U[:,1:3])
kml = km.labels_

plt.figure(2)
plt.scatter(X[:,0],X[:,1],c=kml)
plt.figure(3)
plt.scatter(U[:,1],U[:,2],c=kml)
plt.figure(4)
plt.scatter(U[:,1],U[:,2],c=kml)
plt.figure(5)
plt.scatter(X[:,1],U[:,2],c=kml)
plt.show()
