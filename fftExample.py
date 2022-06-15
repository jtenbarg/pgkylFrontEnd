import numpy as np
import scipy as sp
from utils import gkData
from utils import gkPlot
from utils import polarFFT
import matplotlib.pyplot as plt
import postgkyl as pg
params = {} #Initialize dictionary to store plotting and other parameters
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
plt.rcParams["text.usetex"] = True
#End Preamble##################################


#Tested to handle g0 and g2: VM, 5M, 10M
#Requires a _params.txt file in your data directory of the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting
paramsFile = '/Users/jtenbarg/Desktop/runs/gemEddyv43/Data/gem_params.txt'
paramsFile = '/Users/jtenbarg/Desktop/Runs/ot3D_2D/ot3D_params.txt'

fileNum = 100
suffix = '.bp'
bgDir = 2 #Guide field direction: 0, 1, or 2 = x, y, or z. If 2D data, this direciton must not be in the 2D plane.
tmp = gkData.gkData(paramsFile,fileNum,suffix,'bx',params) #Initialize constants for normalization

#below limits [z0, z1, z2,...] normalized to params["axesNorm"]
params["lowerLimits"] = [-1.e6, -1.e6, -1.e6, -1.e6, -1.e6, -1e6] 
params["upperLimits"] = [ 1.e6,  1.e6,  1.e6,  1.e6,  1.e6,  1.e6]
params["restFrame"] = 1 #Convert pressures to fluid restframe, Pij - n ui uj 


#Define species to normalize and lengths/times 
refSpeciesAxes = 'ion'
refSpeciesTime = 'ion'
speciesIndexAxes = tmp.speciesFileIndex.index(refSpeciesAxes)
speciesIndexTime = tmp.speciesFileIndex.index(refSpeciesTime)
params["axesNorm"] = [tmp.d[speciesIndexAxes], tmp.d[speciesIndexAxes], tmp.d[speciesIndexAxes]]
params["timeNorm"] = tmp.omegaC[speciesIndexTime]


#End input##########################################

bx = gkData.gkData(paramsFile,fileNum,suffix,'bx',params).compactRead() #Tmp read to get grid info

dataShape = np.shape(bx.data)
dims = len(dataShape)
volumeFFT = np.prod(dataShape)

k = [np.fft.fftfreq(dataShape[d], (bx.coords[d][1] - bx.coords[d][0])/(2*np.pi*bx.params["axesNorm"][d])) for d in range(dims)] #Creat wavenumber array

akp, nbin, polar_index, akplim, ebinCorr = polarFFT.initPolar(k, bgDir) #Initialize polar wavenumber bins

#Read and take FFT of data
bx = gkData.gkData(paramsFile,fileNum,suffix,'bx',params).compactRead()
bxFFT = np.fft.fftn(bx.data) * np.sqrt(1/volumeFFT)

#Compute reduced polar spectrum
EbxPol = polarFFT.polarFFTBin(k, bgDir, polar_index, bxFFT)

#Plot spectrum
plt.figure(figsize=(12,8))
plt.loglog(akp,EbxPol,'k', linewidth=2)
plt.autoscale(enable=True, axis='both', tight=True)
plt.xlabel(r'$k_{\perp} \rho_i$')
plt.ylabel('Energy')
plt.autoscale(enable=True, axis='both', tight=True)
plt.show()

