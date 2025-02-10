import numpy as np
from utils import gkData
from utils import auxFuncs
import matplotlib.pyplot as plt
from utils import plotParams


params = {} #Initialize dictionary to store plotting and other parameters
#End preamble####################################

#Tested to handle g0 and g2: VM, 5M, 10M
#Requires a _params.txt file in your data directory of the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting

paramFile = '/Users/jtenbarg/Desktop/Runs/ColbyTurbulence/PKPM_New/Data/pkpm_2d_turb_p2-params.txt'

fileNum = 106 #Frame number
interpFac = 1 #Apply FFT interpolation (for interpFac > 1) of order interpFac.
useB = 0 #Use Bx and By rather than the vector potential. Seems to not work as well
constructJz = 1 #Construct Jz from psi. Less accurate but only requires fields files
saveFig = 0 #Save figure


varid = 'psi' #See table of choices in README
tmp = gkData.gkData(paramFile,fileNum,varid,params) #Initialize constants for normalization

#below limits [z0, z1, z2,...] normalized to params["axesNorm"]
params["lowerLimits"] = [-1e6,  -1e6, -0.e6, -1.e6, -1e6] 
params["upperLimits"] = [1e6,  1e6,  0.e6,  1.e6,  1.e6]
params["restFrame"] = 1 #Convert pressures to fluid restframe, Pij - n ui uj 
params["polyOrderOverride"] = 0 #Override default dg interpolation and interpolate to given number of points


#Define species to normalize and lengths/times 
refSpeciesAxes = 'ion'; refSpeciesAxes2 = 'ion'
refSpeciesTime = 'ion'; 

speciesIndexAxes = tmp.speciesFileIndex.index(refSpeciesAxes); speciesIndexAxes2 = tmp.speciesFileIndex.index(refSpeciesAxes2)
speciesIndexTime = tmp.speciesFileIndex.index(refSpeciesTime)
params["axesNorm"] = [tmp.d[speciesIndexAxes], tmp.d[speciesIndexAxes], tmp.vt[speciesIndexAxes2], tmp.vt[speciesIndexAxes2], tmp.vt[speciesIndexAxes2]]

params["timeNorm"] = tmp.omegaC[speciesIndexTime]
params["axesLabels"] = ['$x/d_i$', '$y/d_i$', '$z/d_p$']

params["plotContours"] = 1 #Overplot contours of the following
params["colorContours"] = 'k' #Color of contours
params["numContours"] = 50 #Number of contours
params["axisEqual"] = 1 #Makes axes equal for 2D
params["symBar"] = 1 #Force colorbar to be symmetric about 0
params["colormap"] = 'bwr'#Colormap for 2D plots: inferno*, bwr (red-blue), any matplotlib colormap


#End input########################################################

var = gkData.gkData(paramFile,fileNum,'psi',params).compactRead()
#Construct jz = -grad^2 psi / mu_0. This minimizes data files that need to be shared to just the fields.
if constructJz:
	[df_dx,df_dy,df_dz] = auxFuncs.genGradient(var.data,var.dx)
	[d2f_dxdx,d2f_dxdy,d2f_dxdz] = auxFuncs.genGradient(df_dx,var.dx)
	[d2f_dydx,d2f_dydy,d2f_dydz] = auxFuncs.genGradient(df_dy,var.dx)
	jz = -(d2f_dxdx + d2f_dydy) / var.mu0
else:
	jz = getattr(gkData.gkData(paramFile,fileNum,'jz',params).compactRead(),'data')

coords0 = var.coords;

if useB:
	bx = getattr(gkData.gkData(paramFile,fileNum,'bx',params).compactRead(),'data')
	by = getattr(gkData.gkData(paramFile,fileNum,'by',params).compactRead(),'data')

if interpFac > 1:
	[psi, coords] = auxFuncs.getFFTInterp(var.data, coords0, fac=interpFac)
	[jz, coords] = auxFuncs.getFFTInterp(jz, coords0, fac=interpFac)
	if useB:
		[bx, coords] = auxFuncs.getFFTInterp(bx, coords0, fac=interpFac)
		[by, coords] = auxFuncs.getFFTInterp(by, coords0, fac=interpFac)
else:
	 psi = var.data; coords = coords0
	
	
x = coords[0]; y = coords[1]; 
dx = [coords[d][1] - coords[d][0] for d in range(2)]

if useB:
	f = bx; g = by
else:
	f = psi; g = None

#Indicies of critical points, X points, and O points (max and min)
critPoints = auxFuncs.getCritPoints(f, g=g, dx=dx)
[xpts, optsMax, optsMin] = auxFuncs.getXOPoints(f, critPoints, g=g, dx=dx)

numC = np.shape(critPoints)[1]
numX = np.shape(xpts)[0]; numOMax = np.shape(optsMax)[0]; numOMin = np.shape(optsMin)[0];

#Create array of 0s with 1s only at X points
binaryMap = np.zeros(np.shape(f)); binaryMap[xpts[:,0],xpts[:,1]] = 1


plt.figure(figsize=(12,8))
plt.pcolormesh(x/params["axesNorm"][0], y/params["axesNorm"][1], np.transpose(jz), shading="gouraud")
plt.plot(x[xpts[:,0]]/params["axesNorm"][0],y[xpts[:,1]]/params["axesNorm"][1],'xk')
plt.plot(x[optsMin[:,0]]/params["axesNorm"][0],y[optsMin[:,1]]/params["axesNorm"][1],'oc')
plt.plot(x[optsMax[:,0]]/params["axesNorm"][0],y[optsMax[:,1]]/params["axesNorm"][1],'om')
plt.xlabel(params["axesLabels"][0]); plt.ylabel(params["axesLabels"][1])
plt.colorbar(); plt.set_cmap(params["colormap"])
if params["symBar"]:
	maxLim = np.max(np.abs(jz))
	plt.clim(-maxLim, maxLim)
if params["plotContours"]:   
    plt.rcParams['contour.negative_linestyle'] = 'solid'
    plt.contour(x/params["axesNorm"][0], y/params["axesNorm"][1], np.transpose(psi),\
        params["numContours"], colors = params["colorContours"], linewidths=0.75)
if params["axisEqual"]:
    plt.gca().set_aspect('equal', 'box')
plt.title('Critical pts = {}, Xpts = {}, OptsMax = {}, OptsMin = {}'.format(numC, numX, numOMax, numOMin) )
if saveFig:
    saveFilename = tmp.filenameBase + 'xPts' + '_interpFac_' + str(interpFac) + '_' + format(fileNum, '04') + '.png'
    plt.savefig(saveFilename, dpi=300)
    print('Figure written to ',saveFilename)
plt.show()

#Plot binaryMap
if False:
	plt.figure(figsize=(12,8))
	plt.pcolormesh(x/params["axesNorm"][0], y/params["axesNorm"][1], np.transpose(binaryMap), shading="gouraud")
	plt.plot(x[xpts[:,0]]/params["axesNorm"][0],y[xpts[:,1]]/params["axesNorm"][1],'xk')

	plt.xlabel(params["axesLabels"][0]); plt.ylabel(params["axesLabels"][1])
	plt.colorbar(); plt.set_cmap('binary')
	plt.gca().set_aspect('equal', 'box')
	if params["plotContours"]:   
	    plt.rcParams['contour.negative_linestyle'] = 'solid'
	    plt.contour(x/params["axesNorm"][0], y/params["axesNorm"][1], np.transpose(psi),\
	        params["numContours"], colors = params["colorContours"], linewidths=0.75)
	plt.show()


