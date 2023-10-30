import numpy as np
from utils import gkData
from utils import gkPlot as plt
#End preamble###################################

#Tested to handle g0 and g2: VM, 5M, 10M
#Requires a _params.txt file in your data directory or the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting
paramFile = '/Users/jtenbarg/Desktop/runs/gemEddyv43/Data/gem_params.txt' 

fileNum = 15
varid = 'ux_elc' #See table of choices in README
params = {} #Initialize dictionary to store plotting and other parameters
tmp = gkData.gkData(paramFile,fileNum,varid,params) #Initialize constants for normalization

params["lowerLimits"] = [-1.e6, -1.e6, -1.e6, -1.e6, -1.e6]
params["upperLimits"] = [1.e6, 1.e6, 1.e6, 1.e6, 1.e6, 1.e6]
params["restFrame"] = 1 #Convert pressures to restframe


#Define species to normalize and lengths/times 
refSpeciesAxes = 'ion'
refSpeciesTime = 'ion'
speciesIndexAxes = tmp.speciesFileIndex.index(refSpeciesAxes)
speciesIndexTime = tmp.speciesFileIndex.index(refSpeciesTime)
params["axesNorm"] = [tmp.d[speciesIndexAxes], tmp.d[speciesIndexAxes], tmp.vt[speciesIndexAxes]]
params["timeNorm"] = tmp.omegaC[speciesIndexTime]
params["axesLabels"] = ['$x/d_p$', '$y/d_p$', '$z/d_p$']
params["timeLabel"] = '$\Omega_{ci}^{-1}$'


params["plotContours"] = 1 #Overplot contours of the following
params["varidContours"] = 'psi' #Plot contours of this variable
params["colorContours"] = 'k' #Color of contours
params["numContours"] = 10 #Number of contours
params["axisEqual"] = 0 #Makes axes equal for 2D
params["symBar"] = 0 #Force colorbar to be symmetric about 0
params["displayTime"] = 1 #Show time of frame
params["colormap"] = 'inferno' #Colormap for 2D plots: inferno*, bwr (red-blue), any matplotlib colormaps
params["absVal"] = 0 #Take absolute value of data
params["log"] = 0 #Take log_10 of data
params["logThresh"] = -4 #Limit lower value of log_10(data)
params["sub0"] = 0 #Subtract data by data(t=0)
params["div0"] = 0 #Divide data by data(t=0)


#End input##################################################
#Let's compute something 
ux = gkData.gkData(paramFile,fileNum,'ux_elc',params)
ux.readData()

bx = gkData.gkData(paramFile,fileNum,'bx',params)
bx.readData()

varGK = ux**2*(ux + bx)/ bx #Do some math

varGK = varGK.integrate(axis=0) #Integrate over first dimension, x
plt.gkPlot(varGK)

#Maybe we want to get the actual coords and data from the Gkeyll object for doing something else
uxData = ux.data
uxCoords = ux.coords




