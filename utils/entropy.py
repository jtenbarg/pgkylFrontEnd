import numpy as np
from utils import gkData
from utils import auxFuncs 


def getEntropy(paramFile,fileNum,spec,params,type=0):
	absVal = params["absVal"]
	params["absVal"] = 0
	f = gkData.gkData(paramFile,fileNum,'dist_'+spec,params).compactRead()
	coords = f.coords
	dimsX = f.dimsX
	dimsV = f.dimsV
	NV = np.shape(f.data);	dimsF = len(NV)
	VInd = list(range(dimsX,dimsF))
	dV = np.prod(f.dx[VInd])
	t = f.time

	if type == 0: #Conventional entropy
		coef = f.data
		fKernel = f.data
	elif type == 1: #Relative entropy
		coef = f.data
		fMax = auxFuncs.equivMax(f, spec)
		fKernel = f.data / fMax
	elif type == 2: #Position space entropy
		n = gkData.gkData(paramFile,fileNum,'n_'+spec,params).compactRead()
		coef = n.data
		fKernel = n.data
	elif type == 3: #Velocity space entropy
		coef = f.data
		n = gkData.gkData(paramFile,fileNum,'n_'+spec,params).compactRead()
		fKernel = f.data / np.transpose([n.data,]*1)

	if absVal:
		s = -coef * np.log(np.abs(fKernel))
	else:
		s = -coef * np.log(fKernel)
	if np.shape(fKernel) == NV:
		s = auxFuncs.integrate(s, dx=f.dx, axis=tuple(VInd))#*dV
		

	return coords[0:dimsX], s, t

