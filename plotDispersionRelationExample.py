import numpy as np
from utils import gkData
import matplotlib.pyplot as plt
from utils import plotParams

#End Preamble############################################

#Requires a _params.txt file in your data directory of the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting
paramsFile = '/Users/jtenbarg/Desktop/Runs/Linear/10m-1-waves_params.txt';

fileNum = 0
varid = 'dispersion'

params = {} #Initialize dictionary to store plotting and other parameters
tmp = gkData.gkData(paramsFile,fileNum,varid,params) #Initialize constants for normalization

#Define species to normalize and lengths/times 
refSpeciesAxes = 'ion'
refSpeciesTime = 'ion'
speciesIndexAxes = tmp.speciesFileIndex.index(refSpeciesAxes)
speciesIndexTime = tmp.speciesFileIndex.index(refSpeciesTime)
params["axesNorm"] = [tmp.rho[speciesIndexAxes], tmp.rho[speciesIndexAxes], tmp.rho[speciesIndexAxes]]
params["timeNorm"] = tmp.omegaC[speciesIndexTime]
params["axesLabels"] = ['$k_x d_p$']
params["timeLabel"] = '$\Omega_{ci}^{-1}$'

#End input#####################################################

var = gkData.gkData(paramsFile,fileNum,varid,params).compactRead()
k = var.coords[0][:]*var.params["axesNorm"][0] #Normalize k
om = var.data/var.params["timeNorm"] #Normalize omega

kMin = min(k)
kMax = max(k)
nModesTot = len(var.data[1])
nModes = int(np.floor(nModesTot/2))+1
colors = plt.cm.jet(np.linspace(0,1,nModes))

plt.figure(figsize=(12,8))
for i in range(nModes): #Plot half of the modes
    plt.plot(k, abs(om[:,i,0]), color=colors[i], linewidth=2)
    plt.plot(k, abs(om[:,i,1]), '--', color=colors[i], linewidth=2)
plt.legend(['Real', 'Imaginary'], frameon=False)
plt.grid()
plt.xlabel(r"$k_x \rho_p$")
plt.ylabel(r"$\omega / \Omega_{cp}$")

plt.gca().set_xlim([kMin, kMax])
plt.show()

