import numpy as np
import scipy as sp
from utils import gkData
from utils import gkPlot
from utils import polarFFT
import matplotlib.pyplot as plt
from utils import plotParams
import postgkyl as pg
from utils import gkPlot
#End Preamble##################################


#Tested to handle g0 and g2: VM, 5M, 10M
#Requires a _params.txt file in your data directory of the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting
paramFile = '/Users/jtenbarg/Desktop/runs/gemEddyv43/Data/gem_params.txt'
paramFile = '/Users/jtenbarg/Desktop/Runs/ot3D_2D/ot3D_params.txt'
paramFile = '/Users/jtenbarg/Desktop/Runs/SULI2022/ot5M_128/ot_g0_5M-params.txt'
paramFile = '/Users/jtenbarg/Desktop/Runs/SULI2022/ot5M_KEP_v1/5m_kep_ot-params.txt'
paramFile = '/Users/jtenbarg/Desktop/Runs/SULI2022/ot_g0_5M_256pi_cfl1/ot_g0_5M-params.txt'
paramFile = '/Users/jtenbarg/Downloads/field-data/pkpm_ot_p1-params.txt'

fileNum = 100
bgDir = 2 #Guide field direction: 0, 1, or 2 = x, y, or z. If 2D data, this direciton must not be in the 2D plane.

params = {} #Initialize dictionary to store plotting and other parameters
tmp = gkData.gkData(paramFile,fileNum,'bx',params) #Initialize constants for normalization

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

bx = gkData.gkData(paramFile,fileNum,'bx',params).compactRead() #Tmp read to get grid info

dataShape = np.shape(bx.data)
dims = len(dataShape)
volumeFFT = np.prod(dataShape)

k = [np.fft.fftfreq(dataShape[d], (bx.coords[d][1] - bx.coords[d][0])/(2*np.pi*bx.params["axesNorm"][d])) for d in range(dims)] #Creat wavenumber array

akp, nbin, polar_index, akplim, ebinCorr = polarFFT.initPolar(k, bgDir) #Initialize polar wavenumber bins

#Read and take FFT of data
bx = gkData.gkData(paramFile,fileNum,'bx',params).compactRead()
by = gkData.gkData(paramFile,fileNum,'by',params).compactRead()
bz = gkData.gkData(paramFile,fileNum,'bz',params).compactRead()

bxFFT = np.fft.fftn(bx.data) * np.sqrt(1/volumeFFT)
byFFT = np.fft.fftn(by.data) * np.sqrt(1/volumeFFT)
bzFFT = np.fft.fftn(bz.data) * np.sqrt(1/volumeFFT)


#Compute reduced polar spectrum
EbxPol = polarFFT.polarFFTBin(k, bgDir, polar_index, bxFFT)
EbyPol = polarFFT.polarFFTBin(k, bgDir, polar_index, byFFT)
EbzPol = polarFFT.polarFFTBin(k, bgDir, polar_index, bzFFT)

#Plot spectrum
plt.figure(figsize=(12,8))
plt.loglog(akp,EbxPol,':r',akp,EbyPol,'-.g',akp,EbzPol,'--b', akp,EbxPol+EbyPol,'k', linewidth=2)
plt.autoscale(enable=True, axis='both', tight=True)
plt.xlabel(r'$k_{\perp} \rho_i$')
plt.ylabel('Energy')
plt.autoscale(enable=True, axis='both', tight=True)
plt.ylim(1e-8, 1e2)
saveFilename = tmp.filenameBase + 'spectrum_' + format(fileNum, '04') + '.png'
plt.savefig(saveFilename, dpi=300)
print('Figure written to ',saveFilename)
plt.show()




