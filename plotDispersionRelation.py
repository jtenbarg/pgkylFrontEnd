import numpy as np
from utils import gkData
from utils import gkPlot as plt
import matplotlib.pyplot as plt

params = {} #Initialize dictionary to store plotting and other parameters

#Include the final underscore or dash in the filename
#Expects a filenameBase + 'params.txt' file to exist! See example_params.txt for the formatting
filenameBase = '/Users/jtenbarg/Downloads/10m-1-waves_';

fileNum = 0
suffix = ''
varid = 'dispersion'

tmp = gkData.gkData(filenameBase,fileNum,suffix,varid,params) #Initialize constants for normalization

#Define species to normalize and lengths/times 
refSpeciesAxes = 'ion'
refSpeciesTime = 'ion'
speciesIndexAxes = tmp.speciesFileIndex.index(refSpeciesAxes)
speciesIndexTime = tmp.speciesFileIndex.index(refSpeciesTime)
params["axesNorm"] = [tmp.d[speciesIndexAxes], tmp.d[speciesIndexAxes], tmp.d[speciesIndexAxes]]
params["timeNorm"] = tmp.omegaC[speciesIndexTime]
params["axesLabels"] = ['$k_x d_p$']
params["timeLabel"] = '$\Omega_{ci}^{-1}$'

#########################################################

var = gkData.gkData(filenameBase,fileNum,suffix,varid,params)
var.readData()


kMax = max(var.coords[0][:])
nModes= len(var.data[1])
plt.figure(figsize=(12,8))
for i in range(int(np.floor(nModes/2))+1):
    plt.plot(var.coords[0][:]*var.params["axesNorm"][0], abs(var.data[:,i,0])/var.params["timeNorm"], 'k', abs(var.data[:,i,1])/var.params["timeNorm"], '--r', linewidth=2)

plt.grid()
plt.xlabel("$k_x d_p$")
plt.ylabel("$\omega_r / \Omega_{cp}$")

plt.gca().set_xlim([0.0, kMax])
plt.show()

