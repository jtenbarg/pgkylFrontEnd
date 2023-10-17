import numpy as np
import matplotlib.pyplot as plt
from utils import plotParams
from utils import gkData
from utils import auxFuncs
#End preamble####################################

#Tested to handle g0 and g2: VM, 5M, 10M
#Requires a _params.txt file in your data directory of the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting
paramsFile = '/Users/jtenbarg/Desktop/runs/gemG0M112x56HiVNoisev2/Data/gem_params.txt';
paramsFile = '/Users/jtenbarg/Desktop/runs/HarrisG0Bg0.1/Data/HarrisBg1_params.txt';
#paramsFile = '/Users/jtenbarg/My Drive/PKPM/64x32x64-vAe-0-08/Data/pkpm_gem_p1-params.txt'

fileNumStart = 0
fileNumEnd = 24
fileSkip = 1
suffix = '.bp'
varid = 'psi' #See table of choices in README
saveFigs = 0

params = {} #Initialize dictionary to store plotting and other parameters
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
n0_ninf = 5; B0_dB = 1; #Used to convert to upstream vA 
va = tmp.vA[specIndex] #Defined by B0 and density in params file
mi = tmp.mu[specIndex]; mu0 = tmp.mu0; n0 = tmp.n[specIndex] #Density in params file
rateFac = B0_dB**2 / va**2 / np.sqrt(mu0*mi*n0) / np.sqrt(n0_ninf)


ts = np.arange(fileNumStart, fileNumEnd+1, fileSkip)
nt = len(ts)

psiX = np.zeros(nt); psiO = np.zeros(nt); t = np.zeros(nt)
for it in range(nt):
	var = gkData.gkData(paramsFile,ts[it],suffix,varid,params).compactRead()
	t[it] = var.time
	saddles = auxFuncs.findSaddles(var.data); nSaddles = len(saddles)
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
	#print(saddles)
#t = t*0.2/params["timeNorm"]
#print(t)
psiDiff = np.abs(psiO - psiX)
dpsidt = np.gradient(psiDiff,t)

plt.figure(figsize=(12,8))
plt.plot(t*params["timeNorm"],dpsidt*rateFac,'k',linewidth=2)
plt.xlabel('$t$' + params["timeLabel"])
plt.ylabel('$E / v_{A} B_0$')
plt.autoscale(enable=True, axis='both', tight=True)
if saveFigs:
    saveFilename = tmp.filenameBase + 'reconRate.png'
    plt.savefig(saveFilename, dpi=300)
    print('Figure written to ',saveFilename)
plt.show()






