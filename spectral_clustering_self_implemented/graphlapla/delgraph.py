import sys
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

def plot_graph(A,p,F):
    n,ts=F.shape
    t = np.random.randint(0,ts,10)

    plt.figure(1)
    for k in range(1,5):
        plt.subplot(2,2,k).axis('off')
        plt.scatter(p[:,0],p[:,1],c=F[:,t[k-1]])
        for i in range(0,n-1):
            for j in range(i+1,n):
                if (A[i,j] != 0):
                    plt.plot([p[i,0],p[j,0]],[p[i,1],p[j,1]],color="gray",linewidth=0.5)

    #plt.figure(1)
    #plt.scatter(p[:,0],p[:,1],c=F[:,1])
    #for i in range(0,n-1):
    #    for j in range(i+1,n):
    #        if (A[i,j] != 0):
    #            plt.plot([p[i,0],p[j,0]],[p[i,1],p[j,1]],color="gray",linewidth=0.5)
    plt.show()

def calc_edges(A):
    n=A.shape[0]
    le = []
    for i in range(0,n-1):
        for j in range(i+1,n):
            if (A[i,j] != 0):
                le.append([i,j])
    ale = np.asarray(le,dtype=np.int32)
    return(ale)


def adj_matrix(Tri_l,n):
    W = np.zeros((n,n))
    for t in Tri_l:
        W[t[0],t[1]]=1
        W[t[1],t[0]]=1
        W[t[0],t[2]]=1
        W[t[2],t[0]]=1
        W[t[1],t[2]]=1
        W[t[2],t[1]]=1

    return(W)

def calc_functions(pts,k):
    n = pts.shape[0]
    mf = np.zeros((n,k))
    for t in range(k):
        i = np.random.randint(0,n)
        dp = np.linalg.norm(pts[:,:]-pts[i,:],axis=1)
        mf[:,t]=np.exp(-(np.linalg.norm(pts[:,:]-pts[i,:],axis=1)**2)/5.0)

    return(mf)

try:
    sys.argv[1]
    sys.argv[2]
    sys.argv[3]
except:
    sys.exit("Invalid number of arguments!! [python3 delgraph.py <input.xy> <output.el> <output.f>]")

p = np.loadtxt(sys.argv[1])
n = p.shape[0]
#pert = np.random.uniform(low=-0.5, high=0.5, size=(n,2))
#points = p + pert
points = p

print('Number of Points: ',points.shape)

tri = Delaunay(points)                         # Delaunay triangulation of points
A = adj_matrix(tri.simplices,points.shape[0])  # compute adjacency matrix (to extract edges)
#le = calc_edges(A)                             # Extract edges
#F = calc_functions(points,100)                 # Generate time slices (second parameter is the number of time-slices)
#plot_graph(A,points,F)
np.savetxt(sys.argv[2],points,fmt='%.6e',delimiter=' ', newline='\n')
np.savetxt(sys.argv[3],A,fmt='%.6e',delimiter=' ', newline='\n')
#np.savetxt(sys.argv[3],F,fmt='%.6f',delimiter=' ', newline='\n')
