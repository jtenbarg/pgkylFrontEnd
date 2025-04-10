import numpy as np
from utils import gkData
from utils import getEnergy
import matplotlib.pyplot as plt
from utils import plotParams

params = {} #Initialize dictionary to store plotting and other parameters
#End preamble####################################

#Note that this file uses the in situ integrated files field-energy and imoms
#Tested to handle g0: VM, PKPM
#Requires a _params.txt file in your data directory of the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting
paramFile = '/Users/jtenbarg/Desktop/Runs/ColbyTurbulence/PKPM_New2/Data/rt_pkpm_2d_turb_p1-params.txt'

varid = '' #See table of choices in README
saveFig = 0
norm0 = 1 #Normalize to Etot(t=0) 
sub0 = 1 #Calculate Delta E WRT t=0


tmp = gkData.gkData(paramFile,0,varid,params) #Initialize constants for normalization

speciesIndexTime = tmp.speciesFileIndex.index('ion') #Time normalization factors
timeNorm = tmp.omegaC[speciesIndexTime]
timeLabel = '$\Omega_{Ci}$'



#End input########################################################

var = gkData.gkData(paramFile,0,varid,params) 

t, fieldEnergy, imoms = getEnergy.getData(var)

E = np.sum(fieldEnergy[:,0:3],axis=1)*var.eps0 / 2. #Electric field
B = np.sum(fieldEnergy[:,3:],axis=1)/(2.*var.mu0) #Magnetic field

nspec = var.nspec; nt = len(t)
T = np.zeros((nspec, nt))
u = np.zeros((nspec, nt))

for s in range(nspec):
	if var.model == 'pkpm':
		T[s,:] = (imoms[s, :, 7] + 2*imoms[s, :, 8]) / 2. #Internal energy
		u[s,:] = np.sum(imoms[s, :, 4:7],axis=1) / 2. #Flow energy
	if var.model == 'vm':
		u[s,:] = np.sum(imoms[s, :, 1:4]**2, axis=1) / (2.*imoms[s, :, 0]) #Flow energy
		T[s,:] = imoms[s,:,4]*var.mu[s]/2. - u[s,:] #Internal energy

Etot = E + B + np.sum(T,axis=0) + np.sum(u,axis=0) #Total energy
E0 = Etot[0]


if sub0:
	ylabel = '$\Delta E$'
	E = E - E[0]
	B = B - B[0]
	Etot = Etot - Etot[0]
	for s in range(nspec):
		T[s,:] = T[s,:] - T[s,0]
		u[s,:] = u[s,:] - u[s,0]
else:
	ylabel = '$E$'

if norm0:
	E = E/E0; B = B/E0; T = T/E0; u = u/E0; Etot = Etot/E0
	ylabel = ylabel + '$ / E_0$'

t = t*timeNorm
cl = ['b','r','m','g']
plt.figure(figsize=(12,8))
plt.plot(t,Etot,'k',t,B,'--k',t,E,'g',linewidth=2)
for s in range(nspec):
    plt.plot(t,T[s,:],'-.'+cl[s],t,u[s,:],'--'+cl[s],linewidth=2)
plt.xlabel('$t$' + timeLabel)
plt.ylabel(ylabel)
plt.legend(['$E_{tot}$','$E_{B}$', '$E_{E}$','$E_{T}$','$E_u$'], frameon=False)
plt.autoscale(enable=True, axis='both', tight=True)
if saveFig:
    saveFilename = tmp.filenameBase + 'energyIntegrated.png'
    plt.savefig(saveFilename, dpi=300)
    print('Figure written to ',saveFilename)
plt.show()
