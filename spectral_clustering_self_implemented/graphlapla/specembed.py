import sys
import numpy as np
import matplotlib.pyplot as plt
import plot_graph as plg

######
# Loading Adjacency Matrix and points
######
A = np.loadtxt(sys.argv[1])
p = np.loadtxt(sys.argv[2])

######
#Computing Laplacian Matrix
######
D = np.diag(np.sum(A,axis=0))
L = D - A

######
# Computing Eigenvalues and Eigenvectors
######
S,U = np.linalg.eigh(L)

######
# Computing Embeding in 2D
######

plg.plot_graph_s(A,p,U[:,1],1)
#plg.plot_graph(A,p,1)
#pspec = np.column_stack((U[:,1],U[:,2]))
#plg.plot_graph(A,pspec,2)

plt.show()
