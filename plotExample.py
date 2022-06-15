import numpy as np
from utils import gkData
from utils import gkPlot as plt
params = {} #Initialize dictionary to store plotting and other parameters
#End Preamble######################

#Tested to handle g0 and g2: VM, 5M, 10M
#Requires a _params.txt file in your data directory or the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting
paramsFile = '/Users/jtenbarg/Desktop/runs/gemEddyv43/Data/gem_params.txt' 

fileNum = 20

suffix = '.bp'
varid = 'bz' #See table of choices in README
tmp = gkData.gkData(paramsFile,fileNum,suffix,varid,params) #Initialize constants for normalization

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
params["axesLabels"] = ['$x/d_p$', '$y/d_p$', '$z/d_p$']
params["timeLabel"] = '$\Omega_{ci}^{-1}$'

params["plotContours"] = 1 #Overplot contours of the following
params["varidContours"] = 'psi' #Plot contours of this variable
params["colorContours"] = 'w' #Color of contours
params["numContours"] = 10 #Number of contours
params["axisEqual"] = 1 #Makes axes equal for 2D
params["symBar"] = 0 #Force colorbar to be symmetric about 0
params["displayTime"] = 1 #Show time of frame
params["colormap"] = 'inferno' #Colormap for 2D plots: inferno*, seismic (red-blue), any matplotlib colormap
params["absVal"] = 0 #Take absolute value of data
params["log"] = 0 #Take log_10 of data
params["logThresh"] = -4 #Limit lower value of log_10(data)
params["sub0"] = 0 #Subtract data by data(t=0)
params["div0"] = 0 #Divide data by data(t=0)

#End input########################################################

var = gkData.gkData(paramsFile,fileNum,suffix,varid,params)
var.readData()

plt.gkPlot(var, show=1, save=0) #show and save are optional. Default show=1, save=0. Saves to var.filenameBase directory

