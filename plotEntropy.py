import numpy as np
from utils import gkData
from utils import entropy
import itertools
import matplotlib.pyplot as plt
from utils import plotParams

params = {} #Initialize dictionary to store plotting and other parameters
#End preamble####################################

#Tested to handle g0 and g2: VP, VM, 5M, 10M
#Requires a _params.txt file in your data directory of the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting
paramFile = '/Users/jtenbarg/Desktop/Runs/LangmuirSarah/landau_damping_1v_33nx-params.txt'

fileNum = 52
spec = 'elc' #See table of choices in README
tmp = gkData.gkData(paramFile,fileNum,'',params) #Initialize constants for normalization
saveFig = 0

#below limits [z0, z1, z2,...] normalized to params["axesNorm"]
params["lowerLimits"] = [-1.e6,  -1e6, -1.e6, -1.e6, -1.e6, -1e6] 
params["upperLimits"] = [1.e6,  1e6,  1.e6,  1.e6,  1.e6,  1.e6]
params["restFrame"] = 0 #Convert pressures to fluid restframe, Pij - n ui uj 
#params["polyOrderOverride"] = 8 #Override default dg interpolation and interpolate to given number of points


#Define species to normalize and lengths/times 
refSpeciesAxes = 'elc'; refSpeciesAxes2 = 'elc'
refSpeciesTime = 'elc'; 

speciesIndexAxes = tmp.speciesFileIndex.index(refSpeciesAxes); speciesIndexAxes2 = tmp.speciesFileIndex.index(refSpeciesAxes2)
speciesIndexTime = tmp.speciesFileIndex.index(refSpeciesTime)
params["axesNorm"] = [tmp.d[speciesIndexAxes], tmp.d[speciesIndexAxes], tmp.d[speciesIndexAxes], tmp.vt[speciesIndexAxes2], tmp.vt[speciesIndexAxes], tmp.vt[speciesIndexAxes]]
#params["axesNorm"] = [tmp.debye[speciesIndexAxes], tmp.vt[speciesIndexAxes], tmp.vt[speciesIndexAxes], tmp.vt[speciesIndexAxes]]

params["timeNorm"] = tmp.omegaC[speciesIndexTime]
params["axesLabels"] = ['$x/d_i$', '$y/d_i$', '$z/d_p$']

params["colormap"] = 'inferno'#Colormap for 2D plots: inferno*, bwr (red-blue), any matplotlib colormap
params["absVal"] = 0 #Take absolute value of data
params["sub0"] = 0 #Subtract data by data(t=0)
params["div0"] = 0 #Divide data by data(t=0)

#End input########################################################

[coords, entropy, t] = entropy.getEntropy(paramFile,fileNum,spec,params,type=0)

axNorm = params["axesNorm"]
plt.figure(figsize=(12,8))
plt.plot(coords[0]/axNorm[0],entropy,'k')
plt.show()





