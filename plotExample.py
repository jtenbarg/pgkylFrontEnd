import numpy as np
from utils import gkData
from utils import gkPlot as plt

params = {} #Initialize dictionary to store plotting and other parameters

#Tested to handle g0 and g2: VM, 5M, 10M
#Include the final underscore or dash in the filename
#Expects a filenameBase + 'params.txt' file to exist! See example_params.txt for the formatting
filenameBase = '/Users/jtenbarg/Desktop/runs/gemEddyv43/Data/gem_' 
#filenameBase = '/Users/jtenbarg/Desktop/runs/PerpShockECDI/Data/p3-maxwell-ions-1x2v_'
#filenameBase = '/Users/jtenbarg/Desktop/Runs/ot3D_2D/ot3D_'
#filenameBase = '/Users/jtenbarg/Research/codes/gkylzero/weibel_4d-'

fileNum = '19'
suffix = '.bp'

tmp = gkData.gkData(filenameBase,fileNum,suffix,params) #Initialize constants for normalization

params["varid"] = 'jz' #See table of choices below
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
params["colormap"] = 'inferno'#Colormap for 2D plots: inferno*, bwr (red-blue), any matplotlib colormap
params["absVal"] = 0 #Take absolute value of data
params["log"] = 0 #Take log_10 of data
params["logThresh"] = -4 #Limit lower value of log_10(data)
params["sub0"] = 0 #Subtract data by data(t=0)
params["div0"] = 0 #Divide data by data(t=0)



##########################################
#Table of choices for varid. i,j, k = x,y,z. spec = species as in filename
#Case is ignored
#Particle distribution: dist_spec
#EM fields: ei or bi
#Scalar potentials: phi, psi 
#Magnitude E or B: mage or magb
#Density: n_spec
#Flow velocity: ui_spec
#Stream function: stream_spec
#Eparallel: epar
#Current: ji
#Species current: ji_spec
#Parallel current: jpar
#Species parallel current: jpar_spec
#J.E, Work: work
#Parallel work, Jpar.Epar: workpar
#J_spec.E, species work: work_spec
#Species parallel work, Jpar_spec.Epar: workpar_spec
#Component work, JiEi: worki
#Species component work, Ji_spec Ei: worki_spec
#Pressure: pij_spec
#Parallel pressure: ppar_spec
#Perp pressure: pperp_spec
#Tr(P): trp_spec
#Temperature: temp_spec
#Parallel temp: temppar_spec
#Perp temp: tempperp_spec
#Tperp / Tpar: tempperppar_spec
#Tpar / Tperp: tempparperp_spec
#Pressure agyrotropy: agyro_spec
#Beta: beta_spec
#Beta par or perp: betapar_spec, betaperp_spec
#Magnetic moment, p_perp_spec / B n_spec: mu_spec
#Gyroradius, sqrt(2 T_perp / m) / (|q| B / m): rho_spec
#Inertial length, c / omega_P: inertiallength_spec
##########################################


var = gkData.gkData(filenameBase,fileNum,suffix,params)
var.readData()

plt.gkPlot(var, show=1, save=0) #show and save are optional. Default show=1, save=0. Saves to filenameBase directory


