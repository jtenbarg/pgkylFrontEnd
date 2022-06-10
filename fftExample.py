import numpy as np
import scipy as sp
from utils import gkData
from utils import gkPlot
from utils import initpolar
from utils import polar_isotropic
import matplotlib.pyplot as plt
import postgkyl as pg
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"


params = {} #Initialize dictionary to store plotting and other parameters

#Tested to handle g0 and g2: VM, 5M, 10M
#Include the final underscore or dash in the filename
#Expects a filenameBase + 'params.txt' file to exist! See example_params.txt for the formatting
filenameBase = '/Users/jtenbarg/Desktop/runs/gemEddyv43/Data/gem_' 

fileNum = 19
suffix = '.bp'
bgDir = 2
tmp = gkData.gkData(filenameBase,fileNum,suffix,'bx',params) #Initialize constants for normalization

#below limits [z0, z1, z2,...] normalized to params["axesNorm"]
params["lowerLimits"] = [-1.e6, -1.e6, -1.e6, -1.e6, -1.e6, -1e6] 
params["upperLimits"] = [ 1.e6,  1.e6,  1.e6,  1.e6,  1.e6,  1.e6]
params["restFrame"] = 1 #Convert pressures to fluid restframe, Pij - n ui uj 


#Define species to normalize and lengths/times 
refSpeciesAxes = 'ion'
refSpeciesTime = 'ion'
speciesIndexAxes = tmp.speciesFileIndex.index(refSpeciesAxes)
speciesIndexTime = tmp.speciesFileIndex.index(refSpeciesTime)
params["axesNorm"] = [tmp.d[speciesIndexAxes], tmp.d[speciesIndexAxes], tmp.vt[speciesIndexAxes], tmp.vt[speciesIndexAxes], tmp.vt[speciesIndexAxes]]
params["timeNorm"] = tmp.omegaC[speciesIndexTime]


#########################################################

bx = gkData.gkData(filenameBase,fileNum,suffix,'bx',params).compactRead() #Tmp read to get grid info

dataShape = np.shape(bx.data)
dims = len(dataShape)
volumeFFT = np.prod(dataShape)

k = [np.fft.fftfreq(dataShape[d], (bx.coords[d][1] - bx.coords[d][0])/(2*np.pi*bx.params["axesNorm"][d])) for d in range(dims)]

#Setup bins and wavenumbers for perpendicular polar spectra
perpind = np.squeeze(np.where(np.arange(dims) != bgDir))
nkp = np.array([dataShape[perpind[d]] for d in range(len(perpind))])
kperp = [k[perpind[d]] for d in range(len(perpind))]
nkpolar = int(np.floor(np.sqrt( np.sum(((nkp)/2)**2) )))
akp, nbin, polar_index, akplim = initpolar.initpolar(nkp[0], nkp[1], 0, kperp[0], kperp[1], [0], nkpolar)
ebinCorr = np.pi*akp/(akp[0]*nbin)

#Read and take FFT of data
bx = gkData.gkData(filenameBase,fileNum,suffix,'bx',params).compactRead()
bxFFT = np.fft.fftn(bx.data) * np.sqrt(1/volumeFFT)

#Compute reduced polar spectra
EbxPol = polar_isotropic.polar_isotropic(nkpolar, dataShape[0], dataShape[1], 0, polar_index, bxFFT, ebinCorr)

#Plot spectrum
plt.figure(figsize=(12,8))
plt.loglog(akp,EbxPol,'k', linewidth=2)
plt.autoscale(enable=True, axis='both', tight=True)
plt.xlabel('$k_\perp d_i$')
plt.ylabel('Energy')
plt.autoscale(enable=True, axis='both', tight=True)
plt.show()

