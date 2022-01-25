from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from matplotlib import cm
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation

"""Il file è composto di 2 parti: 
La prima è il plot delle varie PES 2D, ottenute da una griglia di conti single-point. I risultati con le coppie di coordinate e le varie energie sono storate in risu_pol_0.002.out

L'altra parte è il plot "dinamico" delle traiettorie da un file che contiene tutti i dati (tot_weak_new), vedi file allegato. Tale file è ottenuto dal programma di analisi stat_dyn, e corrisponde al file Stat.pun.file_name. Qui sono collezionate, per ogni time step e per ogni traiettoria, tutte le osservabili (variazione delle coordinate di reazione, energie, popolazioni, etc.) """

"""Iniziamo con la prima parte"""
#Parametri Parte 1 (Plot PES)
fmapname="risu_dia.out" #name of the energy file to build maps.
n_st=4 #number of states involved in the dynamics -> Number of PESs to plot.
#Fine Parametri Parte 1 (Plot PES)

#Parametri parte 2 (plot dinamico)
max_t=1000 #maximum time step for cnnc,nnc.
n_traj=290 #number of trajectories.
ftrajname="tot_weak_new2" #name of the traj files to read angles.
#Fine parametri parte 2 (plot dinamico)


#Sets the list of labels for every single pes
lab=[r"$\left|S_0\right>$",r"$\left|S_1\right>$",r"$\left|S_2\right>$",r"$\left|S_3\right>$",r"$\left|S_4\right>$",r"$\left|S_5\right>$"]

#PARAMETRI FIGURA
fig=plt.figure()
n_row=2 #dimensione griglia subplots, i.e. number of PES involved
n_col=2

fig.subplots_adjust(hspace=0.4, wspace=0.4) #sets spacing
fig.set_size_inches(16.5,12.5,forward=True)
gs=GridSpec(n_row,n_col)
#FINE PARAM FIGURA

"""opens the file of the file """
with open(fmapname,'rb') as f:
       data=np.genfromtxt(f,comments='#')

eref=min(data[:,2])
e,eint=list(),list()


ax=[None]*(n_st)

#BUILDS XY GRIDS TO PLOT MAPS
"the coordinate map is read from the PES file"
#coordinate arrays
X=data[:,0]
Y=data[:,1]
print(np.min(X),np.max(X),np.min(Y),np.max(Y))
#meshes the linear space and the grid
"bisogna creare una griglia di punti in un array bidimensionale facendo il 'mesh' della griglia XY, definendo un nuovo set di coordinate che ricalchino quelle del file"""
xi,yi=np.linspace(np.min(X),np.max(X),180),np.linspace(np.min(Y),np.max(Y),60)
xi,yi=np.meshgrid(xi,yi)

"""interpola i dati X,Y,Energia della pes k-esima sulla griglia xi e yi"""
for k in range(0,n_st):
   e.append(data[:,k+2]) #read energies
   eint.append(scipy.interpolate.griddata((X, Y), e[k], (xi, yi)))#interpolates the values on the energy grids and adds them in the k-th position

"""Riempie i plot delle PESs"""
#fills the energy maps
k=0 #energy index
l=0 #plot index
for j in range(0,n_col):
  for i in range(0,n_row):
     print(i, j)
     ax[l]=fig.add_subplot(gs[i,j])
     ax[l].tick_params(axis='both', which='major', labelsize=15) #specifiche plot: dimensione labels, ticks etc
     isurf=ax[l].imshow((eint[n_st-k-1]-eref)*27.2114,extent=[np.min(xi),np.max(xi),np.min(yi), np.max(yi)],cmap=cm.coolwarm,origin='lower')
     cs=ax[l].contour(xi,yi,(eint[n_st-k-1]-eref)*27.2114,15,colors='black',linewidths=0.2,linestyles='solid') #plotta mappa e contorni
     ax[l].set_xlim(0,180) #range delle x
     ax[l].set_ylim(111,170) #range delle y
     ax[l].plot([],[],lw=0,label=lab[n_st-k-1]) #assegna le legende 
     ax[l].legend(loc="upper left",fontsize=20) #plotta le legende
#     ax[l].clabel(cs, fmt = '%.2f', inline = True)
     ax[l].set_ylabel(r"symNNC $\measuredangle$",size=20)
     ax[l].set_xlabel(r"CNNC $\measuredangle$",size=20)
     l=l+1
     k=k+1
fig.tight_layout()
plt.savefig("pes_static.png",dpi=300)


"""Parte 2: uso Pandas per leggere le traiettorie, identificare traiettorie reattive e non-reattive e associare ogni traiettoria ad un time step ed a una coppia di coordinate reattive"""
##GENERA MEGA-ARRAY CON TUTTE LE DISTRIBUZIONI#
df=pd.read_csv(ftrajname,sep="\s+",header=0) #legge il file delle traiettorie come dataframe. HAI BISOGNO DI UNA PRIMA RIGA CHE DEFINISCA LE "CHIAVI" DI LETTURA DI OGNI COLONNA. Dopodiché potrai chiamare ogni colonna attraverso la sua chiave.
df["symm"]=(df["nnc"].values) #crea una colonna symm dai valori di 'nnc'
scattering,cnnc,cnncnot,nnc,nncnot,scattering_not=[None]*n_st, [None]*n_st, [None]*n_st,[None]*n_st, [None]*n_st,[None]*n_st 
angle,anglenot=[None]*max_t,[None]*max_t #inizializza le liste vuote di lunghezza numero di stati, cosí si possono riempire dentro ai cicli
df["include"]="inc" #inizializza un filtro per le traiettorie

#df.loc[df.time<200,"include"]="inc"
df.loc[(df.cnnc>=80) & (df.cnnc<=130) & (df.istat==3),"include"]="not" #filters the trajectories which end in the middle on the third excited state (numerical instabilities)
#df.loc[(df.time==1000) & (df.istat==1),"include"]="inc"

#sets a label for reactive(non-reactive) trajectories, i.e. cnnc<90 (cnnc>=90) when the time is 1000, the end of the dynamics.
reactive=df.loc[(df["time"].values == max_t) & (df.istat==1)&(df["cnnc"].values<=90), "react"]
not_reactive=df.loc[(df["time"].values == max_t) & (df.istat==1) & (df["cnnc"].values>90), "react"]
condreact=df["react"].isin(reactive)
notreact=df["react"].isin(not_reactive)

for j in range(0,n_st):
#colors the scattering in orange for reactive and turquoise for non-reactives, the x,y of the plot are empty because they will be filled via the animate cycle below"
     ax[j].set_xlim(0,180)
     scattering[j]=ax[n_st-j-1].scatter([],[],c="orange",edgecolor='black',s=20)
     scattering_not[j]=ax[n_st-j-1].scatter([],[],c="paleturquoise",edgecolor='black',s=20)

#animation routine
def animate(i):
   for j in range(0,n_st):
      condt=df["time"].values == i #condition on the time_step i
      condst=df["istat"].values == (j+1) #condition on the state
      condinc=df["include"]=="inc" #condition on the non-filtered trajectories
      cnnc[j]=df.loc[condt &condinc &condst & condreact,"cnnc"].values 
      cnncnot[j]=df.loc[condt&condinc & condst & notreact,"cnnc"].values #assigns cnnc and nnc depending if they are reactive or not, including also the conditions above
      nnc[j]=df.loc[condt & condst &condinc & condreact,"symm"].values
      nncnot[j]=df.loc[condt & condst &condinc & notreact,"symm"].values
      angle[i]=np.vstack((cnnc[j],nnc[j])).T #creates the 2D array of the angles (which uses the same coordinates as the 2D map of the part 1)
      anglenot[i]=np.vstack((cnncnot[j],nncnot[j])).T
      scattering[j].set_offsets(angle[i])
      scattering_not[j].set_offsets(anglenot[i]) #plots the scattering plot, at every time step (condt)

plt.tight_layout()
MovieWriter=animation.writers['ffmpeg']
ani= animation.FuncAnimation(fig,animate,frames=1000,interval=80) #calls the animation function! 
ani.save('z_sc.avi',writer='ffmpeg')#,dpi=200)

plt.show()

