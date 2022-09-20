import numpy as np
from utils import gkData
from utils import gkPlot as plt
from utils import FPC as FPC
import itertools
import matplotlib.pyplot as plt

params = {} #Initialize dictionary to store plotting and other parameters
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
#plt.rcParams["text.usetex"] = True
#End preamble########################


#Tested to handle g0 and g2: VM, 5M, 10M
#Requires a _params.txt file in your data directory of the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting
paramFile = '/Users/jtenbarg/Desktop/runs/gemEddyv43/DataMod/gem_params.txt' 
#paramFile  = '/Users/jtenbarg/Desktop/runs/ECDIKap2D3s3/Data2/ECDI_params.txt'


fileNumStart = 15
fileNumEnd = 15
fileSkip = 1
suffix = '.bp'
varid = ''
spec = 'elc'
nTau = 0 #Frames over which to average. 0 or 1 does no averaging. Note centered ==> nTau must be odd and >= 3 
avgDir = 0 #backwards (-1), forward (1), or centered (0). 

plotAny = 1
plotReducedFPCvsTime = 0 #Must have plotAny = 1
plotReducedFPCatTime1D = -1 #Plot 1v reduced FPCs at this t. Set to -1 to ignore
plotReducedFPCatTime2D = -1 #Plot 1v reduced FPCs at this t. Set to -1 to ignore
saveFigs = 0
showFigs = 1

tmp = gkData.gkData(paramFile,fileNumStart,suffix,varid,params) #Initialize constants for normalization

#below limits [z0, z1, z2,...] normalized to params["axesNorm"]
params["lowerLimits"] = [-2.52e0,  -0.e6, -1.e6, -1.e6, -1.e6, -1e6] 
params["upperLimits"] = [-2.52e0,  0.e6,  1.e6,  1.e6,  1.e6,  1.e6]
params["fieldAlign"] = 0 #Align FPC to the magnetic field. Only use for 3V data.
params["driftAlign"] = 'curvatureDrift' #Align FPC to a drift. Only use for 3V data.
#params["frameXform"] = [1,1,1] #Transform frames, including electric field.

#Define species to normalize and lengths/times 
refSpeciesAxesConf = 'ion'; refSpeciesAxesVel = 'elc'
refSpeciesTime = 'elc'
speciesIndexAxesConf = tmp.speciesFileIndex.index(refSpeciesAxesConf)
speciesIndexAxesVel = tmp.speciesFileIndex.index(refSpeciesAxesVel)
speciesIndexTime = tmp.speciesFileIndex.index(refSpeciesTime)
params["axesNorm"] = [tmp.d[speciesIndexAxesConf], tmp.d[speciesIndexAxesConf], tmp.vt[speciesIndexAxesVel], tmp.vt[speciesIndexAxesVel], tmp.vt[speciesIndexAxesVel]]
params["timeNorm"] = tmp.omegaC[speciesIndexTime]
params["axesLabels"] = ['$x/d_p$', '$y/d_p$', '$v_0/v_t$', '$v_1/v_t$', '$v_2/v_t$']
params["timeLabel"] = '$/ \Omega_{ci}$'
params["colormap"] = 'bwr'#Colormap for 2D plots: inferno*, bwr (red-blue), any matplotlib colormap

##############################################

ts = np.arange(fileNumStart, fileNumEnd+1, fileSkip)
nt = len(ts); t = np.zeros(nt); fpc  = []
for it in range(nt):
	print('Working on frame {0} of {1}'.format(it+1,nt))
	[coords, fpcTmp, t[it]] = FPC.computeFPC(paramFile,ts[it],suffix,spec,params)
	fpc.append(fpcTmp)

	if it==0:
		#Setup x and v grids, dx, and dv
		E = np.atleast_1d(getattr(gkData.gkData(paramFile,ts[it],suffix,'ex',params).compactRead(), 'data'))
		NX = np.shape(E)
		NV = np.shape(fpcTmp[0])
		if len(E) > 1:
			dimsX = len(NX)
			xCoords = coords[0:dimsX]
			dx = []
			dwdt = np.zeros((nt,3,) + NX); dw = np.zeros((nt,3,) + NX);
			for d in range(dimsX-1):
				dx.append(xCoords[d][1] - xCoords[d][0])
		else:
			dimsX = 0; xCoords= [0.]; dx = 1.; dwdt = np.zeros((nt,3)); dw = np.zeros((nt,3));
		dimsF = len(NV); dimsV = dimsF - dimsX
		XInd = list(range(0,dimsX))
		VInd = list(range(dimsX,dimsF))
		vCoords = coords[dimsX:dimsF]
		dv = []
		for d in range(dimsV):
			dv.append(vCoords[d][1] - vCoords[d][0])
		del E
		
		#Find indicies to integrate to reduce to 1V and value of dv for reduced FPCs 
		indCombos1V = list(itertools.combinations(VInd,max(0, dimsV-1)))
		dvCombo1V = []
		for i in range(dimsV):
			p1v = 1.
			for j in range(len(indCombos1V[0])):
				p1v *= dv[indCombos1V[i][j]]

			dvCombo1V.append(p1v)
del fpcTmp

#Perform time average
if nTau > 0:
	fpc = FPC.computeFPCAvg(fpc, nTau, avgDir)

#Compute dw/dt and dw
dwdt = np.zeros((nt,3)); dw = np.zeros((nt,3));
for it in range(nt):
	for d in range(dimsV):
		dwdt[it][d] = np.sum(fpc[it][d],axis=tuple(VInd))*np.prod(dv)

	dw[it] = np.sum(dwdt, axis=0)

dt = 1.
if nt > 1:
	dt = t[1] - t[0]
dw = dw*dt


if plotAny:
	t = t*params["timeNorm"] #Normalize time
	lStyle = [':r', '--g', '-.b', 'k']
	rotate = 0
	figBase = tmp.filenameBase + spec + '_FPC'
	if not (isinstance(params.get('frameXForm'), type(None))):
	    figBase += '_frameXForm'
	    rotate = 1
	if (not (isinstance(params.get('driftAlign'), type(None)))) and not params["fieldAlign"]:
		figBase += '_' + params["driftAlign"]
		rotate = 1
	if params["fieldAlign"]:
		figBase += '_fieldAligned' 
	coordsPlot = coords


	#Plot dw/dt and dw
	plt.figure(figsize=(12,8))
	plt.subplot(121)
	plt.plot(np.sum(dwdt,axis=1),t,'k', linewidth=2)
	for d in range(dimsV):
		plt.plot(dwdt[:,d],t,lStyle[d], linewidth=2)
	plt.plot(np.zeros(nt),t,'k')
	plt.ylabel('$t$' + params["timeLabel"])
	plt.xlabel('$\partial w/\partial t$')
	plt.autoscale(enable=True, axis='both', tight=True)

	plt.subplot(122)
	plt.plot(np.sum(dw,axis=1),t,'k', linewidth=2)
	for d in range(dimsV):
		plt.plot(dw[:,d],t,lStyle[d], linewidth=2)
	plt.plot(np.zeros(nt),t,'k')
	plt.xlabel('$\Delta w$')
	plt.autoscale(enable=True, axis='both', tight=True)
	plt.subplots_adjust(wspace=.0)
	plt.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=1,      # ticks along the bottom edge are off
    right=0,         # ticks along the top edge are off
    labelleft=False) # labels along the bottom edge are off
	if saveFigs:
	    saveFilename = figBase + '_dwdt_dw.png'
	    plt.savefig(saveFilename, dpi=300)
	    print('Figure written to ',saveFilename)
	if showFigs:
		plt.show()

	#Plot 1v reduced FPCs vs time
	if plotReducedFPCvsTime:
		axNorm = params["axesNorm"]; indShift = tmp.dimsX
		titles = ['$C_0$', '$C_1$', '$C_2$' , '$C$']
		for i in range(dimsV+1):
			
			for j in reversed(range(dimsV)):
				d = list(set(VInd) - set(indCombos1V[j]))[0]

				pData = []
				for it in range(nt):
					if i > dimsV-1:
						tmpData = np.sum(fpc[it],axis=0)
						pData.append(np.sum(tmpData,axis=indCombos1V[j])*dvCombo1V[j])
					else:
						pData.append(np.sum(fpc[it][i],axis=indCombos1V[j])*dvCombo1V[j])
				maxData = np.max(np.abs(pData))

				plt.figure(figsize=(12,8))
				c1 = plt.pcolormesh(coordsPlot[d]/axNorm[d+indShift], t, pData, vmin=-maxData, vmax=maxData, cmap = params["colormap"], shading="gouraud")
				plt.xlabel(params["axesLabels"][d+indShift])
				plt.ylabel('$t$' + params["timeLabel"])
				plt.title(titles[i])
				plt.colorbar(c1)
				if saveFigs:
					saveFilename = figBase + '_Red1V_f' + str(i) + '_v' + str(d) + '_frame=' + str(ts[it]) + '.png'
					plt.savefig(saveFilename, dpi=300)
					print('Figure written to ',saveFilename)
				if showFigs:
					plt.show()	
		

	#Plot 1D reduced FPCs at given time
	if plotReducedFPCatTime1D >= 0:
		tplot = plotReducedFPCatTime1D; it = np.searchsorted(ts, tplot)
		axNorm = params["axesNorm"]; indShift = tmp.dimsX
		titles = ['$C_0$', '$C_1$', '$C_2$' , '$C$']
		for i in range(dimsV+1):
			plt.figure(figsize=(12,8))

			for j in range(dimsV):
				d = list(set(VInd) - set(indCombos1V[j]))[0]
				if i > dimsV-1:
					pData = np.sum(fpc[it],axis=0)
					pData = np.sum(pData,axis=indCombos1V[j])*dvCombo1V[j]
				else:
					pData = np.sum(fpc[it][i],axis=indCombos1V[j])*dvCombo1V[j]
				maxData = np.max(np.abs(pData))


				plt.plot(coordsPlot[d]/axNorm[d+indShift], pData,lStyle[d],linewidth=2)
			plt.plot(coordsPlot[d]/axNorm[d+indShift], np.zeros(len(coords[d])),'k',linewidth=1)
			plt.xlabel('$v/v_t$')
			plt.ylabel('$C$')
			plt.title(titles[i])
			plt.autoscale(enable=True, axis='both', tight=True)
			if saveFigs:
				saveFilename = figBase + '_Red1V_f' + str(i) + '_frame=' + str(ts[it]) + '.png'
				plt.savefig(saveFilename, dpi=300)
				print('Figure written to ',saveFilename)
			if showFigs:
				plt.show()	
		

	#Plot 2D FPCs at given time
	if plotReducedFPCatTime2D >= 0 and dimsV >= 2:
		tplot = plotReducedFPCatTime2D; it = np.searchsorted(ts, tplot)
		axNorm = params["axesNorm"]; indShift = tmp.dimsX

		CSub = ['0','1','2', '{tot}']
		for i in range(dimsV+1):
			if dimsV == 3:
				for j in range(dimsV):
					if i > dimsV-1:
						pData = np.sum(fpc[it],axis=0)
						pData = np.sum(pData,axis=j)*dv[j]
					else:
						pData = np.sum(fpc[it][i],axis=j)*dv[j]
					d = list(set(VInd) - set([j]))
				
					maxData = np.max(np.abs(pData))
				
					title = '$C_' + CSub[i] + '($' + params["axesLabels"][d[0]+indShift] + ',' + params["axesLabels"][d[1]+indShift] + '$)$'
					plt.figure(figsize=(12,8))
					c1 = plt.pcolormesh(coordsPlot[d[0]]/axNorm[d[0]+indShift], coordsPlot[d[1]]/axNorm[d[1]+indShift], np.transpose(pData), vmin=-maxData, vmax=maxData, cmap = params["colormap"], shading="gouraud")
					plt.xlabel(params["axesLabels"][d[0]+indShift])
					plt.ylabel(params["axesLabels"][d[1]+indShift])
					plt.colorbar(c1)
					plt.title(title)
					plt.axis('equal')

					if saveFigs:
						saveFilename = figBase + '_Red2V_f' + CSub[i]  + '_v' + str(d) + '_frame=' + str(ts[it]) + '.png'
						plt.savefig(saveFilename, dpi=300)
						print('Figure written to ',saveFilename)
					if showFigs:
						plt.show()
			else:
				if i > dimsV-1:
					pData = np.sum(fpc[it],axis=0)
				else:
					pData = fpc[it][i]
				d = VInd

				maxData = np.max(np.abs(pData))

				title = '$C_' + CSub[i] + '($' + params["axesLabels"][d[0]+indShift] + ',' + params["axesLabels"][d[1]+indShift] + '$)$'
				plt.figure(figsize=(12,8))
				c1 = plt.pcolormesh(coordsPlot[d[0]]/axNorm[d[0]+indShift], coordsPlot[d[1]]/axNorm[d[1]+indShift], np.transpose(pData), vmin=-maxData, vmax=maxData, cmap = params["colormap"], shading="gouraud")
				plt.xlabel(params["axesLabels"][d[0]+indShift])
				plt.ylabel(params["axesLabels"][d[1]+indShift])
				plt.colorbar(c1)
				plt.title(title)
				plt.axis('equal')

				if saveFigs:
					saveFilename = figBase + '_Red2V_f' + CSub[i] + '_v' + str(d) + '_frame=' + str(ts[it]) + '.png'
					plt.savefig(saveFilename, dpi=300)
					print('Figure written to ',saveFilename)
				if showFigs:
					plt.show()
				
		
		if False:
			for j in range(dimsV):
				pData = np.sum(fpc[it],axis=(0,j+1))*dv[j]
				d = list(set(VInd) - set([j]))
				maxData = np.max(np.abs(pData))
				
				title = '$C($' + params["axesLabels"][d[0]+indShift] + ',' + params["axesLabels"][d[1]+indShift] + '$)$'

				plt.figure(figsize=(12,8))
				c1 = plt.pcolormesh(coordsPlot[d[0]]/axNorm[d[0]+indShift], coordsPlot[d[1]]/axNorm[d[1]+indShift], np.transpose(pData), vmin=-maxData, vmax=maxData, cmap = params["colormap"], shading="gouraud")
				plt.xlabel(params["axesLabels"][d[0]+indShift])
				plt.ylabel(params["axesLabels"][d[1]+indShift])
				plt.colorbar(c1)
				plt.title(title)
				plt.axis('equal')
				if saveFigs:
					saveFilename = figBase + '_Red2V_fTot_v' + str(d) + '_frame=' + str(ts[it]) + '.png'
					plt.savefig(saveFilename, dpi=300)
					print('Figure written to ',saveFilename)
				if showFigs:
					plt.show()
		elif False:
			pData = np.sum(fpc[it],axis=0)
			d = VInd
			maxData = np.max(np.abs(pData))
				
			title = '$C($' + params["axesLabels"][d[0]+indShift] + ',' + params["axesLabels"][d[1]+indShift] + '$)$'

			plt.figure(figsize=(12,8))
			c1 = plt.pcolormesh(coordsPlot[d[0]]/axNorm[d[0]+indShift], coordsPlot[d[1]]/axNorm[d[1]+indShift], np.transpose(pData), vmin=-maxData, vmax=maxData, cmap = params["colormap"], shading="gouraud")
			plt.xlabel(params["axesLabels"][d[0]+indShift])
			plt.ylabel(params["axesLabels"][d[1]+indShift])
			plt.colorbar(c1)
			plt.title(title)
			plt.axis('equal')
			if saveFigs:
				saveFilename = figBase + '_Red2V_fTot_v' + str(d) + '_frame=' + str(ts[it]) + '.png'
				plt.savefig(saveFilename, dpi=300)
				print('Figure written to ',saveFilename)
			if showFigs:
				plt.show()
				

				
		

















