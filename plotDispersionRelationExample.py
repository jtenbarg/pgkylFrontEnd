import numpy as np
from utils import gkData
from utils import gkPlot as plt
import matplotlib.pyplot as plt
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
params["axesNorm"] = [tmp.rho[speciesIndexAxes], tmp.rho[speciesIndexAxes], tmp.rho[speciesIndexAxes]]
params["timeNorm"] = tmp.omegaC[speciesIndexTime]
params["axesLabels"] = ['$k_x d_p$']
params["timeLabel"] = '$\Omega_{ci}^{-1}$'

#########################################################

var = gkData.gkData(filenameBase,fileNum,suffix,varid,params)
var.readData()


kMax = max(var.coords[0][:])/var.params["axesNorm"][0]
nModes= len(var.data[1])
plt.figure(figsize=(12,8))
for i in range(int(np.floor(nModes/2))+1):
    plt.plot(var.coords[0][:]/var.params["axesNorm"][0], abs(var.data[:,i,0])*var.params["timeNorm"], 'k', abs(var.data[:,i,1])*var.params["timeNorm"], '--r', linewidth=2)

plt.grid()
plt.xlabel(r"$k_x \rho_p$")
plt.ylabel("$\omega_r / \Omega_{cp}$")

plt.gca().set_xlim([0.0, kMax])
plt.show()

