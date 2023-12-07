import numpy as np
from utils import gkData
from utils import gkPlot as plt
params = {} #Initialize dictionary to store plotting and other parameters
#End preamble####################################

#Tested to handle g0 and g2: VM, 5M, 10M, VP, PKPM
#Requires a _params.txt file in your data directory of the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting
paramFile = '/Users/jtenbarg/My Drive/PKPM/64x32x64-vAe-0-08/Data/pkpm_gem_p1-params.txt'

fileNumStart = 0
fileNumEnd = 185
fileSkip = 1
varid = 'tempperppar_ion' #See table of choices in README
showFigs = 0
saveFigs = 1
tmp = gkData.gkData(paramFile,fileNumStart,varid,params) #Initialize constants for normalization

#below limits [z0, z1, z2,...] normalized to params["axesNorm"]
params["lowerLimits"] = [-1.e6, -1.e6, -1.e6, -1.e6, -1.e6, -1e6] 
params["upperLimits"] = [1.e6,  1.e6,  1.e6,  1.e6,  1.e6,  1.e6]
params["restFrame"] = 1 #Convert pressures to fluid restframe, Pij - n ui uj 
#params["polyOrderOverride"] = 8 #Override default dg interpolation and interpolate to given number of points


#Define species to normalize and lengths/times 
refSpeciesAxes = 'ion'; refSpeciesAxes2 = 'elc'
refSpeciesTime = 'ion'

speciesIndexAxes = tmp.speciesFileIndex.index(refSpeciesAxes); speciesIndexAxes2 = tmp.speciesFileIndex.index(refSpeciesAxes2)
speciesIndexTime = tmp.speciesFileIndex.index(refSpeciesTime)
params["axesNorm"] = [tmp.d[speciesIndexAxes], tmp.d[speciesIndexAxes], tmp.d[speciesIndexAxes], tmp.vt[speciesIndexAxes], tmp.vt[speciesIndexAxes]]
#params["axesNorm"] = [tmp.rho[speciesIndexAxes], tmp.vt[speciesIndexAxes2], tmp.d[speciesIndexAxes], tmp.vt[speciesIndexAxes], tmp.vt[speciesIndexAxes]]

params["timeNorm"] = tmp.omegaC[speciesIndexTime]
params["axesLabels"] = [r'$x/d_i$', r'$y/d_i$', '$z/d_p$']
params["timeLabel"] = '$\Omega_{ci}^{-1}$'

params["plotContours"] = 1 #Overplot contours of the following
params["varidContours"] = 'psi' #Plot contours of this variable
params["colorContours"] = 'w' #Color of contours
params["numContours"] = 10 #Number of contours
params["axisEqual"] = 1 #Makes axes equal for 2D
params["displayTime"] = 0 #Show time of frame
params["symBar"] = 0 #Force colorbar to be symmetric about 0
params["colormap"] = 'inferno'#Colormap for 2D plots: inferno*, bwr (red-blue), any matplotlib colormap
params["absVal"] = 0 #Take absolute value of data
params["log"] = 0 #Take log_10 of data
params["logThresh"] = -4 #Limit lower value of log_10(data)
params["sub0"] = 0 #Subtract data by data(t=0)
params["div0"] = 0 #Divide data by data(t=0)

#End input########################################################

ts = np.arange(fileNumStart, fileNumEnd+1, fileSkip)
nt = len(ts)

for it in range(nt):
	var = gkData.gkData(paramFile,ts[it],varid,params).compactRead()
	plt.gkPlot(var, show=showFigs, save=saveFigs) #show and save are optional. Default show=1, save=0. Saves to filenameBase directory

