import numpy as np
import scipy as sp
from utils import gkData

#Handle arbitrary dimensionality for numpy gradient method
def genGradient(var, dx):

    dims = len(dx)
    ax = tuple(np.arange(dims))
    shape0 = np.shape(var)
    dvardx = np.zeros(shape0); dvardy = np.zeros(shape0); dvardz = np.zeros(shape0)
    
    if dims==3:
        [dvardx, dvardy, dvardz] = np.gradient(var,dx[0],dx[1],dx[2],edge_order=2,axis=ax)
    elif dims==2:
        [dvardx, dvardy] = np.gradient(var,dx[0],dx[1],edge_order=2,axis=ax)
    else:
        dvardx = np.gradient(var,dx[0],edge_order=2,axis=ax)

    return dvardx, dvardy, dvardz

#Handle arbitrary dimensionality for numpy gradient method
def eigthOrderGrad2D(f, dx):
    shapeF = np.shape(f); dims = len(shapeF)
    if dims > 2 and shapeF[2] == 1:
        f = np.squeeze(f); shapeF = np.shape(f)
    nrows, ncols = shapeF
    dfdx = np.zeros((nrows, ncols))
    dfdy = np.zeros((nrows, ncols))
    dfdz = np.zeros((nrows, ncols))
    [dfdx, dfdy] = np.gradient(f,dx[0],dx[1],edge_order=2)
    for i in range(4, nrows-4):
        for j in range(4, ncols-4):
            dfdy[i,j] = (-f[i, j+4]/280.0 + f[i, j+3]*4.0/105.0 - f[i, j+2]/5.0 + f[i, j+1]*4.0/5.0 \
                - f[i, j-1]*4.0/5.0 + f[i, j-2]/5.0 - f[i, j-3]*4.0/105.0 + f[i, j-4]/280.0) / (dx[1])
            dfdx[i,j] = (-f[i+4, j]/280.0 + f[i+3, j]*4.0/105.0 - f[i+2, j]/5.0 + f[i+1, j]*4.0/5.0 \
                - f[i-1, j]*4.0/5.0 + f[i-2, j]/5.0 - f[i-3, j]*4.0/105.0 + f[i-4, j]/280.0) / (dx[0])
    if dims == 3:
        dfdx = dfdx[...,np.newaxis]; dfdy = dfdy[...,np.newaxis]; dfdz = dfdz[...,np.newaxis]

    return dfdx, dfdy, dfdz


#Handle arbitrary dimensionality for numpy gradient method
def eigthOrderGrad2Dv2(f, dx):
    shapeF = np.shape(f); dims = len(shapeF)
    if dims > 2 and shapeF[2] == 1:
        f = np.squeeze(f); shapeF = np.shape(f)
    nrows, ncols = shapeF
    dfdx = np.zeros((nrows, ncols))
    dfdy = np.zeros((nrows, ncols))
    dfdz = np.zeros((nrows, ncols))
    [dfdx, dfdy] = np.gradient(f,dx[0],dx[1])

    a = 1./280.; b = 4./105.; c = 1./5.; d = 4./5.
    for i in range(4, nrows-4):
        for j in range(4, ncols-4):
            dfdy[i,j] = (-a*f[i, j+4] + b*f[i, j+3] - c*f[i, j+2] + d*f[i, j+1] \
                - d*f[i, j-1] + c*f[i, j-2] - b*f[i, j-3] + a*f[i, j-4]) / (dx[1])
            dfdx[i,j] = (-a*f[i+4, j] + b*f[i+3, j] - c*f[i+2, j] + d*f[i+1, j] \
                - d*f[i-1, j] + c*f[i-2, j] - b*f[i-3, j] + a*f[i-4, j]) / (dx[0])
    if dims == 3:
        dfdx = dfdx[...,np.newaxis]; dfdy = dfdy[...,np.newaxis]; dfdz = dfdz[...,np.newaxis]

    return dfdx, dfdy, dfdz


#Compute and return rotation matrix for two provided vectors, B and V
#Basis is b, v x b, and b x (v x b)
def rotMatrixArbitrary(B,V):

    #Handle case for B = (1,0,0)
    if B[1] == 0 and B[2] == 0:
        B[1] = B[0]*1e-24

    rot = np.zeros((3,3))
    magB = np.linalg.norm(B); b = B/magB
    magV = np.linalg.norm(V); v = V/magV

    vxb1 = v[1]*b[2] - v[2]*b[1]
    vxb2 = v[2]*b[0] - v[0]*b[2]
    vxb3 = v[0]*b[1] - v[1]*b[0]

    bxvxb1 = b[1]*vxb3 - b[2]*vxb2
    bxvxb2 = b[2]*vxb1 - b[0]*vxb3
    bxvxb3 = b[0]*vxb2 - b[1]*vxb1

    #Rotation Matrix must be Orthonormal
    mag2=np.sqrt(vxb1**2. + vxb2**2. + vxb3**2)
    mag3=np.sqrt(bxvxb1**2. + bxvxb2**2. + bxvxb3**2)

    #Setup Rotation Matrix
    rot[0] = b

    rot[1,0]=vxb1/mag2
    rot[1,1]=vxb2/mag2
    rot[1,2]=vxb3/mag2

    rot[2,0]=bxvxb1/mag3
    rot[2,1]=bxvxb2/mag3
    rot[2,2]=bxvxb3/mag3

    return rot

#Compute and return rotation matrix to align with B
#Basis is b, (1,0,0) x b, and b x ((1,0,0) x b)
def rotMatrixFieldAligned(B):

    #Handle case for B = (1,0,0)
    if B[1] == 0 and B[2] == 0:
        B[1] = B[0]*1.e-24

    rot = np.zeros((3,3)); 
    magB = np.linalg.norm(B); b = B/magB

    #Rotation Matrix must be Orthonormal
    magB2=np.sqrt(b[2]**2.+b[1]**2.)
    magB3=np.sqrt((b[2]**2.+b[1]**2.)**2.+b[0]*b[1]*b[0]*b[1]+b[0]*b[2]*b[0]*b[2])

    #Setup Rotation Matrix
    rot[0] = b

    rot[1,0]=0.
    rot[1,1]=-b[2]/magB2
    rot[1,2]=b[1]/magB2

    rot[2,0]=(b[1]**2.+b[2]**2.)/magB3
    rot[2,1]=-b[0]*b[1]/magB3
    rot[2,2]=-b[0]*b[2]/magB3

    return rot

#Returns the list of all saddle points of the input matrix
def findSaddles(mat : np.ndarray) -> list:

    (N, M) = mat.shape

    jMax = np.argmax(mat, axis = 1) # index of col for max in each row
    iMin = np.argmin(mat, axis = 0) # index of row for min in each col

    IJMax = [(i,jMax[i]) for i in range(N)] # list of indexes of max of each row
    IJMin = [(iMin[j],j) for j in range(M)] # list of indexes of min of each col

    maxRowMinCol = list(set(IJMax) & set(IJMin)) # max of row, min of col


    iMax = np.argmax(mat, axis = 0) # index of row for max in each col
    jMin = np.argmin(mat, axis = 1) # index of col for min in each row

    IJMax = [(iMax[j],j) for j in range(M)] # list of indexes of max of each col
    IJMin = [(i,jMin[i]) for i in range(N)] # list of indexes of min of each row

    minRowMaxCol = list(set(IJMax) & set(IJMin)) # min of row, max of col


    return maxRowMinCol + minRowMaxCol

#Returns an equivalent Maxwellian, fMax = n / (2 pi vth^2)^(0.5*dimsV) * exp(-(v-u)^2 / 2 vth^2)
def equivMax(f, spec):
    coords = f.coords
    dimsX = f.dimsX
    dimsV = f.dimsV
    NV = np.shape(f.data);  dimsF = len(NV)
    VInd = list(range(dimsX,dimsF))
    specIndex = f.speciesFileIndex.index(spec)
    dens = gkData.gkData(f.paramFile,f.fileNum,'n_'+spec,f.params).compactRead()
    temp = gkData.gkData(f.paramFile,f.fileNum,'temp_'+spec,f.params).compactRead()
    dirs = ['x', 'y', 'z']
    u = []      
    for i in range(dimsV):
        u.append(np.atleast_1d(getattr(gkData.gkData(f.paramFile,f.fileNum,'u' + dirs[i]+'_'+spec,f.params).compactRead(), 'data')))

    vth = np.sqrt(temp.data/temp.mu[specIndex])
    fMax = np.empty_like(f.data)
    expo = 0.5*dimsV
    nRange = np.prod(NV)
    for i in range(nRange):
        ind = np.unravel_index(i, NV)
        v2 = 0.
        for iv in VInd:
            v2 += (coords[iv][ind[iv]] - u[iv-dimsX][ind[0:dimsX]])**2.
    
        vth2 = vth[ind[0:dimsX]]**2.
        fMax[ind] = (dens.data[ind[0:dimsX]] / (2*np.pi*vth2)**expo) * np.exp(-v2/(2*vth2))

    return fMax

#Generalized Simpson's rule for arbitrary dimensions.
def integrate(f, dx=1., axis=None):
    dims = len(np.shape(f)); 
    if axis is None:
        axis = tuple(np.arange(0,dims))
    if type(dx) is not list:
        dx = dx*np.ones(dims)
    tot = f
    if type(axis) is int:
        tot = sp.integrate.simps(tot, dx=dx[axis], axis=axis)
    else:
        axisSorted = sorted(axis, reverse=True)
        for d in axisSorted:
            tot = sp.integrate.simps(tot, dx=dx[d], axis=d)

    return tot


