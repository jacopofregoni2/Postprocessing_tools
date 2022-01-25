from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from matplotlib import cm
from scipy.interpolate import griddata


"""Iniziamo con la prima parte"""
#Parametri Parte 1 (Plot PES)
fmapname="risu_dia.out" #name of the energy file to build maps.
n_st=2 #number of states involved in the dynamics -> Number of PESs to plot.
#Fine Parametri Parte 1 (Plot PES)

#Sets the list of labels for every single pes
lab=[r"$\left|S_0\right>$",r"$\left|S_1\right>$",r"$\left|S_2\right>$",r"$\left|S_3\right>$",r"$\left|S_4\right>$",r"$\left|S_5\right>$"]

#PARAMETRI FIGURA
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

#FINE PARAM FIGURA

"""opens the file of the file """
with open(fmapname,'rb') as f:
       data=np.genfromtxt(f,comments='#')

eref=min(data[:,2])
e=list()


#BUILDS XY GRIDS TO PLOT MAPS
"the coordinate map is read from the PES file"
"qui pero' i punti delle PESs sono equispaziati, quindi nel tuo caso dovrai prima trovare la funzione delle PESs"
#coordinate arrays
X=data[:,0]
Y=data[:,1]
#meshes the linear space and the grid
xi,yi=np.linspace(np.min(X),np.max(X),180),np.linspace(np.min(Y),np.max(Y),60)
xi,yi=np.meshgrid(xi,yi)


cmlist=[cm.coolwarm,cm.cividis_r]
"""interpola i dati X,Y,Energia della pes k-esima sulla griglia xi e yi"""
for k in range(0,n_st):
   e.append(data[:,k+2]) #read energies
   eint=scipy.interpolate.griddata((X, Y), e[k]-eref, (xi, yi))#interpolates the values on the energy grids and adds them in the k-th position
   ax.plot_surface(xi,yi,eint,cmap=cmlist[k])

#xs=[180,0.167,7.22]
#ys=[115.356,169,123.314]
#zs=[0,3.38912836,0.8814373]
#ax.scatter(xs,ys,zs,c='firebrick',alpha=1)

ax.set_ylabel(r"symNNC $\measuredangle$",size=20)
ax.set_xlabel(r"CNNC $\measuredangle$",size=20)
fig.tight_layout()
plt.savefig("pes_static.png",dpi=300)

plt.show()

