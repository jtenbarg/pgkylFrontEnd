import numpy as np
from utils import gkData
from utils import auxFuncs 

def between(val, x):
	dx = x[1]-x[0]
	for i in range(len(x)):
		#if x[i] >= val:
		if x[i]+dx/2. >= val:
			break

	return i

def computeFPC(paramFile,fileNum,spec,params):
	rotate = 0
	dirs = ['x', 'y', 'z']
	E = []
	B = []
	
	for i in range(3):
		E.append(np.atleast_1d(getattr(gkData.gkData(paramFile,fileNum,'e' + dirs[i],params).compactRead(), 'data')))
		B.append(np.atleast_1d(getattr(gkData.gkData(paramFile,fileNum,'b' + dirs[i],params).compactRead(), 'data')))
	fTmp = gkData.gkData(paramFile,fileNum,'dist_'+spec,params).compactRead()
	if not (isinstance(params.get('useDeltaF'), type(None))):
		if params["useDeltaF"]:
			f0 = gkData.gkData(paramFile,0,'dist_'+spec,params).compactRead()
			fTmp = fTmp - f0

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

	if False:
		dens = gkData.gkData(paramFile,fileNum,'n_'+spec,params).compactRead()
		temp = gkData.gkData(paramFile,fileNum,'temp_'+spec,params).compactRead()
		print(NV)
		vth = np.sqrt(temp.data/temp.mu[specIndex])
		fMax = np.empty_like(f)
		for ivx in range(NV[0]):
			for ivy in range(NV[1]):
				for ivz in range(NV[2]):
					v2 = vCoords[0][ivx]**2. + vCoords[0][ivy]**2. + vCoords[0][ivz]**2.
					fMax[ivx,ivy,ivz] = (dens.data / (2*np.pi*vth*vth)**1.5) * np.exp(-v2/(2*vth*vth))
		f = f - fMax

	#Compute dfdv components
	dfdv = []
	for i in VInd:
		dfdv.append(np.gradient(f, dx[i],  axis=i ))
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
			drift.append(np.atleast_1d(getattr(gkData.gkData(paramFile,fileNum,driftType + dirs[i] + '_' + spec,params).compactRead(), 'data')))
		rot = np.zeros(NX + (3,3))
		for i in range(np.prod(NX)):
			ind = np.unravel_index(i, NX)
			Bl = []; [Bl.append(B[ix][ind]) for ix in range(3)]
			driftl = []; [driftl.append(drift[ix][ind]) for ix in range(3)]
			rot[ind] = auxFuncs.rotMatrixArbitrary(np.squeeze(Bl),driftl)
		rotate = 1

	#Compute rotation matrix for field aligned coords. Overrides drift alignment
	if params["fieldAlign"]:
		rot = np.zeros(NX + (3,3))
		for i in range(np.prod(NX)):
			ind = np.unravel_index(i, NX)
			Bl = []; [Bl.append(B[ix][ind]) for ix in range(3)]
			#rot[ind] = auxFuncs.rotMatrixFieldAligned(np.squeeze(Bl))
			rot[ind] = auxFuncs.rotMatrixArbitrary(np.squeeze(Bl),[0,0,1])
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
			indL = list(ind); 
			for iv in VInd:
				indL[iv] = int(iRot[iv - dimsX])
			ind = tuple(indL); 
		v2 = np.linalg.norm(v)**2
		for iv in VInd:
			fpc[iv-dimsX][ind] = fpc[iv-dimsX][ind] + FPCPreFac*v2*dfdvLoc[iv-dimsX]*E[iv-dimsX][ind[0:dimsX]]

	
	return coords, fpc, t

def computeFPCPKPM(paramFile,fileNum,spec,params):
	bb_gradu = gkData.gkData(paramFile,fileNum,'bbgradu_' + spec,params).compactRead()

	fTmp = gkData.gkData(paramFile,fileNum,'dist0_'+spec,params).compactRead()

	if not (isinstance(params.get('useDeltaF'), type(None))):
		if params["useDeltaF"]:
			f0 = gkData.gkData(paramFile,0,'dist0_'+spec,params).compactRead()
			fTmp = fTmp - f0

	f = fTmp.data
	coords = fTmp.coords
	dx = fTmp.dx
	t = fTmp.time
	specIndex = fTmp.speciesFileIndex.index(spec)
	PreFac =  0.5*fTmp.mu[specIndex]
	del fTmp
	NV = np.shape(f)

	du_dx_weight = np.zeros(NV)
	vf = np.zeros(NV)
	for i in range(0, NV[-1]):
		du_dx_weight[..., i] =  bb_gradu.data[...]*coords[-1][i]*coords[-1][i] #vpar^2 bb grad u
		vf[..., i] = f[..., i]*coords[-1][i] # vpar f
	du_dx_weight = du_dx_weight * PreFac

	#fpc = 0.5*m_s vpar^2 (bb:grad u) d (vpar f) / d vpar
	fpc = du_dx_weight*np.gradient(vf, dx[-1], edge_order=2, axis=-1)
	
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
			winU = min(it +1+ win, nt)
		elif avgDir == 1:
			winL = it
			winU = min(it+1+ winMax, nt)	
		else:
			winL = it; winU = it+1
			print("Unrecognized summing direction! avgDir must be one of -1,0,1.")
		FPCout.append(np.mean(FPCin[winL:winU],axis=0))

	return FPCout

#Initialize polar, (vperp, vguide), grid and bins for polar FPC
def polarBin(v, bgDir, FPCMatrix):

    dims = np.shape(v)[0]
    
    if dims == 1: 
        avp = [] 
        nbin = 0
        polar_index = []
        avplim = []
    else:
        #Setup bins and wavenumbers for perpendicular polar spectra
        perpind = np.squeeze(np.where(np.arange(dims) != bgDir))
        nvp = np.array([len(v[d]) for d in range(len(perpind))])
        vperp = [v[perpind[d]] for d in range(len(perpind))]
        nvpolar = int(np.floor(np.sqrt( np.sum(((nvp)/2)**2) )))
        nvx = nvp[0]
        nvy = nvp[1]
        
        nbin = np.zeros(nvpolar) #Number of vx,vy in each polar bins
        polar_index = np.zeros((nvx, nvy), dtype=int) #Polar index to simplify binning 
        if nvx == 1 & nvy==1:
            dvp = 0
        elif nvx == 1:
            dvp = vperp[1][1] - vperp[1][0]
        elif nvy == 1:
            dvp = vperp[0][1] - vperp[0][0]
        else:
            dvp = max(vperp[0][1] - vperp[0][0], vperp[1][1] - vperp[1][0])        
        avp = dvp/2 + (np.linspace(0, nvpolar, nvpolar))*dvp #vperp grid centers
        avplim = (np.linspace(0,nvpolar, nvpolar+1))*dvp #Bin limits
        #Re-written to avoid loops. Necessary for large grids.
        [vxg, vyg] = np.meshgrid(vperp[1],vperp[0]) #Deal with meshgrid weirdness (so do not have to transpose)
        vp = np.sqrt(vxg**2 + vyg**2)
        pn  = np.where(vp >= avplim[nvpolar])
        polar_index[pn[0], pn[1]] = nvpolar-1  
        nbin[nvpolar-1] = nbin[nvpolar-1] + len(pn[0])
        for iv in range(0, nvpolar):
            pn = np.where((vp < avplim[iv+1]) & (vp >= avplim[iv]))
            polar_index[pn[0], pn[1]] = iv
            nbin[iv] = nbin[iv] + len(pn[0])
        
        shape = np.shape(FPCMatrix)
        FPCMatrixPolar = np.zeros((nvpolar, shape[bgDir]))
        for i in range(0, nvp[0]):
            for j in range(0, nvp[1]):
                for k in range(0, shape[bgDir]):
                    ivperp = polar_index[i,j]
                    FPCMatrixPolar[ivperp,k] = FPCMatrixPolar[ivperp,k] + 2*np.pi*avp[ivperp]*FPCMatrix[k,i,j]/nbin[ivperp]#newFPCMatrix[i,j,k]
    
    return avp, nbin, polar_index, avplim, FPCMatrixPolar

#Convert cartesian FPC to polar FPC and bin.
def polarBinOld(v, bgDir, polar_index, FPCMatrix):
    
    dims = np.shape(v)[0]
    perpind = np.squeeze(np.where(np.arange(dims) != bgDir))
    nvp = np.array([len(v[d]) for d in range(len(perpind))])
    nvpolar = int(np.floor(np.sqrt( np.sum(((nvp)/2)**2) )))

    if dims == 1: #Nothing to do
        FPCMatrixPolar = [] 
    elif dims == 2: #Assumes in perp plane
        FPCMatrixPolar = np.zeros(nvpolar)
    
        for i in range(0, nvp[0]):
            for j in range(0, nvp[1]):
                if not (i == 0 and j == 0):
                    ivperp = polar_index[i,j]   
                    FPCMatrixPolar[ivperp] = FPCMatrixPolar[ivperp] + FPCMatrix[i,j]
    else:#Bins 3D FPC matrix in Par-Perp1-Perp2 coords into 2D Perp-Par FPC matrix
        shape = np.shape(FPCMatrix)
        FPCMatrixPolar = np.zeros((nvpolar, shape[bgDir]))

        
        for i in range(0, nvp[0]):
            for j in range(0, nvp[1]):
                for k in range(0, shape[bgDir]):
                    ivperp = polar_index[i,j]
                    FPCMatrixPolar[ivperp,k] = FPCMatrixPolar[ivperp,k] + FPCMatrix[k,i,j]
        
    return FPCMatrixPolar





