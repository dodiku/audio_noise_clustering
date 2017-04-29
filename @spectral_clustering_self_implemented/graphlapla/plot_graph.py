import numpy as np
import matplotlib.pyplot as plt

# simple plot of a graph
def plot_graph(A,p,pn):
    n=A.shape[0]
    plt.figure(pn)
    plt.scatter(p[:,0],p[:,1])
    for i in range(0,n-1):
        for j in range(i+1,n):
            if (A[i,j] != 0):
                plt.plot([p[i,0],p[j,0]],[p[i,1],p[j,1]],color="blue")

# plot of a graph coloring by a function on the nodes
def plot_graph_s(A,p,cm,pn):
    n=A.shape[0]
    plt.figure(pn)
    for i in range(0,n-1):
        for j in range(i+1,n):
            if (A[i,j] != 0):
                plt.plot([p[i,0],p[j,0]],[p[i,1],p[j,1]],color="black",linewidth=0.5,alpha=0.5)

    sc1=plt.scatter(p[:,0],p[:,1],c=cm,s=50,edgecolor='none',alpha=1.0,cmap='rainbow')
    plt.colorbar(sc1)

def plot_graph_h(A,p,cm):
    cm_max = np.amax(cm)
    cm_min = np.amin(cm)
    xmax = np.amax(p[:,0])
    xmin = np.amin(p[:,0])
    ymax = np.amax(p[:,1])
    ymin = np.amin(p[:,1])
    dx = xmax-xmin
    dy = ymax-ymin
    #print(cm_max,cm_min)
    #print(xmax,xmin,ymax,ymin)
    s = np.sqrt(dx*dx+dy*dy)
    n,n=A.shape
    plt.figure(1)
    for i in range(0,n-1):
        for j in range(i+1,n):
            if (A[i,j] != 0):
                plt.plot([p[i,0],p[j,0]],[p[i,1],p[j,1]],color="black",linewidth=0.5)

    plt.scatter(p[:,0],p[:,1],s=0.8,edgecolor='none')
    for i in range(0,n):
        hcm = s*((cm[i]-cm_min)/(cm_max-cm_min)) - s/2
        print(cm[i],hcm)
        plt.plot([p[i,0],p[i,0]],[p[i,1],(p[i,1]+hcm)],color="blue")
    plt.show()

def plot_specf(x,f,pn):
    plt.figure(pn)
    plt.plot(x,f,color='black')
    zr = np.zeros((x.shape[0],1))
    plt.scatter(x,zr,s=0.5)
    #plt.fill_between(x, 0, f)

def plot_spec(s,pn):
    plt.figure(pn)
    x = range(s.shape[0])
    plt.plot(x,s,'*',color='black')
