import numpy as np
import matplotlib.pyplot as plt
from utils import gkData
from utils import findSaddles
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
#End preamble####################################

#Tested to handle g0 and g2: VM, 5M, 10M
#Requires a _params.txt file in your data directory of the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting
paramsFile = '/Users/jtenbarg/Desktop/runs/gemG0M224x112Noise/Data/gem_params.txt';

fileNumStart = 0
fileNumEnd = 30
fileSkip = 1
suffix = '.bp'
varid = 'psi' #See table of choices in README
tmp = gkData.gkData(paramsFile,fileNumStart,suffix,varid,params) #Initialize constants for normalization

#below limits [z0, z1, z2,...] normalized to params["axesNorm"]
params["lowerLimits"] = [-1.e6, -1.e0, -1.e6, -1.e6, -1.e6, -1e6] 
params["upperLimits"] = [1.e6,  1.e0,  1.e6,  1.e6,  1.e6,  1.e6]
params["restFrame"] = 1 #Convert pressures to fluid restframe, Pij - n ui uj 


#Define species to normalize and lengths/times 
refSpeciesAxes = 'ion'
refSpeciesTime = 'ion'
speciesIndexAxes = tmp.speciesFileIndex.index(refSpeciesAxes)
speciesIndexTime = tmp.speciesFileIndex.index(refSpeciesTime)
params["axesNorm"] = [tmp.d[speciesIndexAxes], tmp.d[speciesIndexAxes], tmp.d[speciesIndexAxes], tmp.vt[speciesIndexAxes], tmp.vt[speciesIndexAxes]]
params["timeNorm"] = tmp.omegaC[speciesIndexTime]
params["axesLabels"] = ['$y/d_p$', '$z/d_p$', '$z/d_p$']
params["timeLabel"] = '$\Omega_{ci}$'

params["plotContours"] = 1 #Overplot contours of the following
params["varidContours"] = 'psi' #Plot contours of this variable
params["colorContours"] = 'w' #Color of contours
params["numContours"] = 10 #Number of contours
params["axisEqual"] = 0 #Makes axes equal for 2D
params["symBar"] = 0 #Force colorbar to be symmetric about 0
params["displayTime"] = 1 #Show time of frame
params["colormap"] = 'inferno'#Colormap for 2D plots: inferno*, bwr (red-blue), any matplotlib colormap
params["absVal"] = 0 #Take absolute value of data
params["log"] = 0 #Take log_10 of data
params["logThresh"] = -4 #Limit lower value of log_10(data)
params["sub0"] = 0 #Subtract data by data(t=0)
params["div0"] = 0 #Divide data by data(t=0)

#End input########################################################


#Converts rate to E/vA_perp delta B = E / (vA/c)^2 * (B0/delta B)^2
specIndex = tmp.speciesFileIndex.index('ion');
n0_ninf = 5; B0_dB = 1; va_c = tmp.vA[specIndex] * np.sqrt(n0_ninf)/ tmp.c
rateFac = B0_dB**2 / va_c**2

ts = np.arange(fileNumStart, fileNumEnd+1, fileSkip)
nt = len(ts)

psiX = np.zeros(nt); psiO = np.zeros(nt); t = np.zeros(nt)
for it in range(nt):
	var = gkData.gkData(paramsFile,ts[it],suffix,varid,params).compactRead()
	t[it] = var.time
	saddles = findSaddles.saddles(var.data); nSaddles = len(saddles)
	if nSaddles == 0:
		print('Found no saddle points in frame ' + str(it))
	elif nSaddles == 1:
		psiX[it] = var.data[saddles[0]]
	else:
		print('Warning, found ' + str(nSaddles) + ' saddle points in frame ' + str(it) + '. Choosing first non-edge saddle.')
		nn = np.shape(var.data)
		for sad in saddles:
			if (sad[0] != 0 and sad[0] != nn[0]) and (sad[1] != 0 and sad[1] != nn[1]):
				psiX[it] = var.data[sad]
				break
	
	psiO[it] = var.max

psiDiff = np.abs(psiO - psiX)
#dpsidt = np.diff(psiDiff) / np.diff(t)
dpsidt = np.gradient(psiDiff,t)

plt.figure(figsize=(12,8))
plt.plot(t*params["timeNorm"],dpsidt*rateFac,'k',linewidth=2)
plt.xlabel('$t$' + params["timeLabel"])
plt.ylabel('$E / v_{A} B_0$')
plt.autoscale(enable=True, axis='both', tight=True)
plt.show()






