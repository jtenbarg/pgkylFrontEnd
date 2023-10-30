import numpy as np
from utils import gkData
import matplotlib.pyplot as plt
from utils import plotParams

#End preamble########################

#This file computes the change in energies from the moment and fields files.
#Tested to handle g0 and g2: VM, 5M, 10M
#Requires a _params.txt file in your data directory of the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting
paramsFile = '/Users/jtenbarg/Desktop/runs/gemG0M112x56HiVNoisev2/Data/gem_params.txt';
paramsFile = '/Users/jtenbarg/Desktop/Runs/PKPM/Langmuir2/Data/pkpm_landau_damping_p1-params.txt';
paramsFile = '/Users/jtenbarg/Desktop/Runs/PKPM/AWk1_1DHi/Data/pkpm_alf_wave_p1-params.txt';
paramsFile = '/Users/jtenbarg/Desktop/Runs/PKPM/OT2DPKPMv1/Data/pkpm_ot_p1-params.txt';
paramsFile = '/Users/jtenbarg/Desktop/Runs/PKPM/OT2DVM/Data/ot-g0-jcp-param_params.txt';
paramsFile = '/Users/jtenbarg/My Drive/PKPM/64x32x64-vAe-0-08/Data/pkpm_gem_p1-params.txt'
paramsFile = '/Users/jtenbarg/Desktop/Runs/PKPM/AWk1_1DHi2/Data/pkpm_alf_wave_p1-params.txt';
paramsFile = '/Users/jtenbarg/Desktop/runs/HarrisG0Bg0.1/Data/HarrisBg1_params.txt';

fileNumStart = 0
fileNumEnd = 24
fileSkip = 1
sub0 = 1
saveFigs = 1

varid = ''
params = {} #Initialize dictionary to store plotting and other parameters
tmp = gkData.gkData(paramsFile,fileNumStart,varid,params) #Initialize constants for normalization

#below limits [z0, z1, z2,...] normalized to params["axesNorm"]
params["lowerLimits"] = [-1.e6, -1.e6, -1.e6, -1.e6, -1.e6, -1e6] 
params["upperLimits"] = [ 1.e6,  1.e6,  1.e6,  1.e6,  1.e6,  1.e6]
params["restFrame"] = 0 #Convert pressures to fluid restframe, Pij - n ui uj 


#Define species to normalize and lengths/times 
refSpeciesAxes = 'ion'
refSpeciesTime = 'ion'
speciesIndexAxes = tmp.speciesFileIndex.index(refSpeciesAxes)
speciesIndexTime = tmp.speciesFileIndex.index(refSpeciesTime)
#params["axesNorm"] = [tmp.d[speciesIndexAxes], tmp.d[speciesIndexAxes], tmp.vt[speciesIndexAxes], tmp.vt[speciesIndexAxes], tmp.vt[speciesIndexAxes]]
params["timeNorm"] = tmp.omegaC[speciesIndexTime]
params["axesLabels"] = ['$x/d_p$', '$y/d_p$', '$z/d_p$']
params["timeLabel"] = '$\Omega_{ci}$'

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

#End input###############################################

nspec = len(tmp.mu)
ts = np.arange(fileNumStart, fileNumEnd+1, fileSkip)
nt = len(ts)

E = np.zeros(nt)
B = np.zeros(nt)
J = np.zeros(nt)
Work = np.zeros((nspec+1,nt))
P = np.zeros((nspec, nt))
u = np.zeros((nspec, nt))
t = np.zeros(nt)
params["restFrame"] = 0
for i in range(nt):
    
    ex = gkData.gkData(paramsFile,ts[i],'ex',params).compactRead()
    ey = gkData.gkData(paramsFile,ts[i],'ey',params).compactRead()
    ez = gkData.gkData(paramsFile,ts[i],'ez',params).compactRead()
    t[i] = ex.time*ex.params["timeNorm"]
    
    bx = gkData.gkData(paramsFile,ts[i],'bx',params).compactRead()
    by = gkData.gkData(paramsFile,ts[i],'by',params).compactRead()
    bz = gkData.gkData(paramsFile,ts[i],'bz',params).compactRead()
    E2 = (ex**2 + ey**2 + ez**2)*ex.eps0 / 2.
    B2 = (bx**2 + by**2 + bz**2)/ 2. / bx.mu0
    E[i] = getattr(E2.integrate(), 'data')
    B[i] = getattr(B2.integrate(), 'data')


    jx = gkData.gkData(paramsFile,ts[i],'jx',params).compactRead()
    jy = gkData.gkData(paramsFile,ts[i],'jy',params).compactRead()
    jz = gkData.gkData(paramsFile,ts[i],'jz',params).compactRead()
    J2 = jx**2 + jy**2 + jz**2
    JE = jx*ex + jy*ey + jz*ez
    
    J[i] =  getattr(J2.integrate(), 'data')
    
    Work[0, i] =  getattr(JE.integrate(), 'data')
    
    for s in range(nspec):
        varid = 'n_' + tmp.speciesFileIndex[s]
        n = gkData.gkData(paramsFile,ts[i],varid,params).compactRead()
        if n.model == '5m':
            varid = 'pxx_' + tmp.speciesFileIndex[s]
            tmp =  gkData.gkData(paramsFile,ts[i],varid,params).compactRead()
        else:
            varid = 'trp_' + tmp.speciesFileIndex[s]
            tmp = (gkData.gkData(paramsFile,ts[i],varid,params).compactRead())/2.
                
        P[s,i] = getattr(tmp.integrate(), 'data')   
        varid = 'ux_' + tmp.speciesFileIndex[s]
        ux = gkData.gkData(paramsFile,ts[i],varid,params).compactRead()
        varid = 'uy_' + tmp.speciesFileIndex[s]
        uy = gkData.gkData(paramsFile,ts[i],varid,params).compactRead()
        varid = 'uz_' + tmp.speciesFileIndex[s]
        uz = gkData.gkData(paramsFile,ts[i],varid,params).compactRead()
        tmp = n*(ux**2 + uy**2 + uz**2)*n.mu[s]/2
        u[s,i] = getattr(tmp.integrate(), 'data')
        
        JE = n*(ux*ex+uy*ey+uz*ez)
        Work[s+1, i] =  n.q[s]*getattr(JE.integrate(), 'data')

#t=ts
Etot = np.sum(P,axis=0) + E + B
if n.model == 'pkpm':
    Etot = Etot + np.sum(u,axis=0)
    T = P
else:
    T = P - u

E0 = Etot[0]
dT = T/E0
dT0 = T/E0
dP = P/E0
dB = B/E0
dE = E/E0
du = u/E0
dEtot = Etot/E0
if sub0:
    yLabel = '$\Delta E / E_0$'
    dE = dE - dE[0]
    dB = dB - dB[0]
    dEtot = dEtot - dEtot[0]
    for s in range(nspec):
        dT[s,:] = dT[s,:] - dT[s,0]
        dT0[s,:] = (T[s,:] - T[s,0]) / T[s,0]
        dP[s,:] = dP[s,:] - dP[s,0]
        du[s,:] = du[s,:] - du[s,0]
else:
    yLabel = 'Energy'


cl = ['b','r','m','g']
plt.figure(figsize=(12,8))
plt.plot(t, dEtot, 'k', t, dB, '--k', t, dE, 'g', linewidth=2)
for s in range(nspec):
    plt.plot(t,dT[s,:],'-.'+cl[s],t,du[s,:],'--'+cl[s],linewidth=2)
    #plt.plot(t,dT[s,:],'-.r',t,du[s,:],'--b',linewidth=2)
plt.plot(t,np.zeros(nt),'k')
plt.legend(['$E_{tot}$','$E_{B}$', '$E_{E}$','$E_{T}$','$E_u$'], frameon=False)
plt.xlabel('$t$' + tmp.params["timeLabel"])
plt.ylabel(yLabel)
plt.autoscale(enable=True, axis='both', tight=True)
if saveFigs:
    saveFilename = tmp.filenameBase + 'Energy_pgkyl' + '.png'
    plt.savefig(saveFilename, dpi=300)
    print('Figure written to ',saveFilename)
plt.show()


plt.figure(figsize=(12,8))
plt.plot(t,J,'k', linewidth=2)
plt.xlabel('$t$' + tmp.params["timeLabel"])
plt.ylabel('$J^2$')
plt.autoscale(enable=True, axis='both', tight=True)
if saveFigs:
    saveFilename = tmp.filenameBase + 'J2_pgkyl' + '.png'
    plt.savefig(saveFilename, dpi=300)
    print('Figure written to ',saveFilename)
plt.show()

plt.figure(figsize=(12,8))
plt.plot(t,Work[0,:],'k', linewidth=2)
leg = ['$J \cdot E$']
for s in range(nspec):
    plt.plot(t,Work[s+1,:],'--'+cl[s],linewidth=2)
    leg.append('$J_{' + tmp.speciesFileIndex[s][0] + '} \cdot E$')
plt.legend(leg, frameon=False)
plt.plot(t,np.zeros(nt),'k')
plt.xlabel('$t$' + tmp.params["timeLabel"])
plt.ylabel('$J\cdot E$')
plt.autoscale(enable=True, axis='both', tight=True)
if saveFigs:
    saveFilename = tmp.filenameBase + 'Work_pgkyl' + '.png'
    plt.savefig(saveFilename, dpi=300)
    print('Figure written to ',saveFilename)
plt.show()

leg = []
plt.figure(figsize=(12,8))
for s in range(nspec):
    plt.plot(t,dT0[s,:]*2/3,'-'+cl[s],linewidth=2)
    leg.append('$E_{T_' + tmp.speciesFileIndex[s][0] + '}$')
plt.plot(t,np.zeros(nt),'k')
plt.legend(leg, frameon=False)
plt.xlabel('$t$' + tmp.params["timeLabel"])
plt.ylabel('$\Delta T_s/T_{0s}$')
plt.autoscale(enable=True, axis='both', tight=True)
if saveFigs:
    saveFilename = tmp.filenameBase + 'dTemperature_pgkyl' + '.png'
    plt.savefig(saveFilename, dpi=300)
    print('Figure written to ',saveFilename)
plt.show()

leg = []
plt.figure(figsize=(12,8))
for s in range(nspec):
    plt.plot(t,T[s,:]*2/3,'-'+cl[s],linewidth=2)
    leg.append('$T_' + tmp.speciesFileIndex[s][0] + '$')
plt.plot(t,np.zeros(nt),'k')
plt.legend(leg, frameon=False)
plt.xlabel('$t$' + tmp.params["timeLabel"])
plt.ylabel('$T$')
plt.autoscale(enable=True, axis='both', tight=True)
if saveFigs:
    saveFilename = tmp.filenameBase + 'Temperature_pgkyl' + '.png'
    plt.savefig(saveFilename, dpi=300)
    print('Figure written to ',saveFilename)
plt.show()
