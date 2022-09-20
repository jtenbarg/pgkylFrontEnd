import numpy as np
from utils import gkData
from utils import rotMatrix 

def between(val, x):
	for i in range(len(x)):
		if x[i] >= val:
			break

	return i

def computeFPC(paramFile,fileNum,suffix,spec,params):
	rotate = 0
	dirs = ['x', 'y', 'z']
	E = []
	B = []
	
	for i in range(3):
		E.append(np.atleast_1d(getattr(gkData.gkData(paramFile,fileNum,suffix,'e' + dirs[i],params).compactRead(), 'data')))
		B.append(np.atleast_1d(getattr(gkData.gkData(paramFile,fileNum,suffix,'b' + dirs[i],params).compactRead(), 'data')))

	fTmp = gkData.gkData(paramFile,fileNum,suffix,'dist_'+spec,params).compactRead()
	f = fTmp.data
	coords = fTmp.coords
	dx = fTmp.dx
	t = fTmp.time
	specIndex = fTmp.speciesFileIndex.index(spec)
	FPCPreFac =  -0.5*fTmp.q[specIndex]
	del fTmp


	NX = np.shape(E[0])
	NV = np.shape(f)
	if len(E[0]) > 1:
		dimsX = len(NX)
		xCoords = coords[0:dimsX]	
	else:
		dimsX = 0; xCoords= [0.]; 
	dimsF = len(NV)
	XInd = list(range(0,dimsX))
	VInd = list(range(dimsX,dimsF))
	vCoords = coords[dimsX:dimsF]
	#Compute dfdv components
	dfdv = []
	for i in VInd:
		dfdv.append(np.gradient(f, dx[i], edge_order=2, axis=i ))
	for i in range(3 - len(VInd)):
		dfdv.append(np.zeros(NV)) #Ensure dfdv is 3-vector


	#Transform E to appropriate frame based on user supplied velocities
	#Untested
	if not (isinstance(params.get('frameXForm'), type(None))):
		xformVel = params["frameXForm"]
		[xformVel.append(0) for i in range(3 - len(xformVel))] #Ensure it is a 3-vector

		#E' = E + u x B 
		E[0] = E[0] + xformVel[1]*B[2] - xformVel[2]*B[1]
		E[1] = E[1] + xformVel[2]*B[0] - xformVel[0]*B[2]
		E[2] = E[2] + xformVel[0]*B[1] - xformVel[1]*B[0]

		for iv in VInd:
			coords[iv] = coords[iv] - xformVel[iv - dimsX]

	#Read apprproiate drift for rotation into drift aligned coords
	if not (isinstance(params.get('driftAlign'), type(None))):
		driftType = params["driftAlign"]
		drift = []
		for i in range(3):
			drift.append(np.atleast_1d(getattr(gkData.gkData(paramFile,fileNum,suffix,driftType + dirs[i] + '_' + spec,params).compactRead(), 'data')))
		rot = np.zeros(NX + (3,3))
		for i in range(np.prod(NX)):
			ind = np.unravel_index(i, NX)
			Bl = []; [Bl.append(B[ix][ind]) for ix in range(3)]
			driftl = []; [driftl.append(drift[ix][ind]) for ix in range(3)]
			rot[ind] = rotMatrix.arbitrary(np.squeeze(Bl),driftl)
		rotate = 1

	#Compute rotation matrix for field aligned coords. Overrides drift alignment
	if params["fieldAlign"]:
		rot = np.zeros(NX + (3,3))
		for i in range(np.prod(NX)):
			ind = np.unravel_index(i, NX)
			Bl = []; [Bl.append(B[ix][ind]) for ix in range(3)]
			rot[ind] = rotMatrix.fieldAligned(np.squeeze(Bl))
		rotate = 1

	#Rotate electric field
	if rotate:
		for i in range(np.prod(NX)):
			ind = np.unravel_index(i, NX)
			El = []; [El.append(E[ix][ind]) for ix in range(3)]
			Etmp = np.matmul(rot[ind],El)
			for ix in range(3):
				E[ix][ind] = Etmp[ix]

	dfdvFlat = []
	for i in range(3):
		dfdvFlat.append(dfdv[i].flatten())
	fpc = np.zeros((3,) + NV)
	v = np.zeros(3); iRot = np.zeros(3);
	nRange = np.prod(NV)
	for i in range(nRange):
		ind = np.unravel_index(i, NV)
		for iv in VInd:
			v[iv - dimsX] = coords[iv][ind[iv]]
		dfdvLoc = []; [dfdvLoc.append(dfdvFlat[ix][i]) for ix in range(3)]
		if rotate:
			v = np.squeeze(np.matmul(rot[ind[0:dimsX]],v)) #v = vpar, vperp1, vperp2
			for ix in range(3):
				iRot[ix] = between(v[ix], vCoords[0])
			dfdvLoc = np.squeeze(np.matmul(rot[ind[0:dimsX]],dfdvLoc))
			indL = list(ind)
			for iv in VInd:
				indL[iv] = int(iRot[iv - dimsX])
			ind = tuple(indL)
		v2 = np.linalg.norm(v)**2
		for iv in VInd:
			fpc[iv-dimsX][ind] = fpc[iv-dimsX][ind] + FPCPreFac*v2*dfdvLoc[iv-dimsX]*E[iv-dimsX][ind[0:dimsX]]

	
	return coords, fpc, t

def computeFPCAvg(FPCin, nTau, avgDir):
	dimsFPC = np.shape(FPCin)
	nt = dimsFPC[0]

	FPCout = []; winMax = nTau - 1
	for it in range(nt):
		if avgDir == -1:
			winL = max(0,it - winMax)
			winU = it+1	
		elif avgDir == 0:
			win = int(np.floor(winMax / 2))
			winL = max(0,it - win)
			winU = min(it + 1+ win, nt)
		elif avgDir == 1:
			winL = it
			winU = min(it+1 + winMax, nt)	
		else:
			winL = it; winU = it+1
			print("Unrecognized summing direction! avgDir must be one of -1,0,1.")
			
		FPCout.append(np.mean(FPCin[winL:winU],axis=0))

	return FPCout





