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
def integrateSimpson(f, dx=1., axis=None):
    dims = len(np.shape(f)); 
    if axis is None:
        axis = tuple(np.arange(0,dims))
    if (type(dx) is not list) and (not isinstance(dx,np.ndarray)):
        dx = dx*np.ones(dims)
    tot = f
    if type(axis) is int:
        tot = sp.integrate.simpson(tot, dx=dx[axis], axis=axis)
    else:
        axisSorted = sorted(axis, reverse=True)
        for d in axisSorted:
            tot = sp.integrate.simpson(tot, dx=dx[d], axis=d)

    return tot

#Simple sum over arbitrary dimensions.
def integrate(f, dx=1., axis=None):
    dims = len(np.shape(f)); 
    if axis is None:
        axis = tuple(np.arange(0,dims))
    if (type(dx) is not list) and (not isinstance(dx,np.ndarray)):
        dx = dx*np.ones(dims)
    tot = f
    if type(axis) is int:
        tot = np.sum(tot, axis=axis)*dx[axis]
    else:
        axisSorted = sorted(axis, reverse=True)
        for d in axisSorted:
            tot = np.sum(tot, axis=d)*dx[d]

    return tot

#Interpolate 2D using zero padded FFT. Must be square data!
def getFFTInterp(f, x, fac=2): 
    n0 = np.shape(f);
    #n = [n0[d]*fac for d in range(2)]
    pad_width = int( (fac*n0[0] - n0[0]) / 2.  )
  
    ff = np.fft.fft2(f)  
    ff_shift = np.fft.fftshift(ff)
    ff_shift_pad = np.pad(ff_shift, pad_width=pad_width, mode='constant', constant_values=0)
    ff_pad = np.fft.ifftshift(ff_shift_pad)
    g = np.fft.ifft2(ff_pad)*fac*fac #Rescaling is correct for even grids  
    n = np.shape(g)

    dx = [x[d][1] - x[d][0] for d in range(2)]
    xL = [x[d][0] - dx[d]/2. for d in range(2)] #recover upper and lower cell edges from centers
    xU = [x[d][-1] + dx[d]/2. for d in range(2)]
    coords = [np.linspace(xL[d], xU[d], n[d] + 1) for d in range(2)] #new grid
    coords = [0.5*(coords[d][:-1] + coords[d][1:]) for d in range(2)] #shift to cell centers

    return g.real, coords

#Find unique critical points
def getCritPoints(f, g=None, dx=[1.,1.]):
    n = np.shape(f);
    
    if g is None:        
        [df_dx,df_dy,df_dz] = genGradient(f,dx) #Used if input is psi
    else:
        df_dx = -g; df_dy = f #Used if input is Bx and By

    iGrad = np.zeros(n)

    #x and y points one cell ahead, considering periodic BCs
    jx = np.arange(1,n[0]); jx = np.append(jx, 0)
    jy = np.arange(1,n[1]); jy = np.append(jy, 0)

    tmpDx = np.zeros(4); tmpDy = np.zeros(4); 
    iGradx = np.sign(df_dx); iGrady = np.sign(df_dy)
    for ix in range(n[0]):
        ixp = jx[ix]
        for iy in range(n[1]):
            iyp = jy[iy]

            tmpDx[0]=iGradx[ix,iy]
            tmpDx[1]=iGradx[ixp,iy]
            tmpDx[2]=iGradx[ix,iyp]
            tmpDx[3]=iGradx[ixp,iyp]

            tmpDy[0]=iGrady[ix,iy]
            tmpDy[1]=iGrady[ixp,iy]
            tmpDy[2]=iGrady[ix,iyp]
            tmpDy[3]=iGrady[ixp,iyp]  
        
            
            #Check is point is a critical point
            if np.all(tmpDx == tmpDx[0]) or np.all(tmpDy == tmpDy[0]):
                iGrad[ix,iy] = 0
            else:
                iGrad[ix,iy] = 1

            if np.any(tmpDx == 0.) or np.any(tmpDy == 0.):
                iGrad[ix,iy] = 1
    

    critPoints = np.where(iGrad == 1)
    numC = np.shape(critPoints)[1]

    dxx = dx[0]; dyy = dx[1];
    ndegen = 0
    #Find sub-cell extrema of candidate critical points and eliminate non-maximal critical points
    for ip in range(numC):
        ix = critPoints[0][ip];  iy = critPoints[1][ip]
        ixp = jx[ix]; iyp = jy[iy]
        Px1=df_dx[ix,iy]
        Px2=df_dx[ixp,iy]
        Px3=df_dx[ixp,iyp]
        Px4=df_dx[ix,iyp]

        Py1=df_dy[ix,iy]
        Py2=df_dy[ixp,iy]
        Py3=df_dy[ixp,iyp]
        Py4=df_dy[ix,iyp]

        AA=(Px1 - Px4)*(Py2 - Py3) - (Px2 - Px3)*(Py1 - Py4)
        BB=dyy*(2.*Px2*Py1 - Px3*Py1 - 2.*Px1*Py2 + Px4*Py2 + Px1*Py3 - Px2*Py4)
        CC=dyy*dyy*(Px1*Py2 - Px2*Py1)

        tmp=BB*BB-4.*AA*CC

        if tmp >= 0:
            y1 = (-BB - np.sqrt(tmp) )/(2.*AA)
            x1 = dxx*( dyy*Px1 + y1*(Px4 - Px1) )/( dyy*(Px1 - Px2) + y1*( Px2 - Px1 - Px3 + Px4) )
            y2 = (-BB + np.sqrt(tmp) )/(2.*AA)
            x2 = dxx*( dyy*Px1 + y2*(Px4 - Px1) )/( dyy*(Px1 - Px2) + y2*( Px2 - Px1 - Px3 + Px4) )
            
            tmp2 = 0.; tmp3 = 0.
            if (y1 >= 0.) and (x1 >= 0.) and (y1 <= dyy) and (x1 <= dxx):
                x01 = x1
                y01 = y1
                tmp2 = 1.
            if (y2 >= 0.) and (x2 >= 0.) and (y2 <= dyy) and (x2 <= dxx):
                x02 = x2
                y02 = y2
                tmp3 = 1.
            if (y1 >= 0.) and (x1 >= 0.) and (y2 >= 0.) and (x2 >= 0.) and (y1 <= dyy) and (x1 <= dxx) and (y2 <= dyy) and (x2 <= dxx):
                #print('Degenerate point found.')
                ndegen += 1
                iGrad[ix,iy] = 0
            tmp4 = tmp2 + tmp3
            if tmp4 == 0.:
                iGrad[ix,iy] = 0
           

        else:
            iGrad[ix,iy] = 0

    if ndegen > 0:
        print('Degenerate points found', ndegen)    
    critPoints = np.where(iGrad == 1)

    print('Critical points found ', np.shape(critPoints)[1])
    return critPoints




#Compute the Hessian matrix
def getHessian(f, g=None, dx=[1.,1.]):
    if g is None:
        [df_dx,df_dy,df_dz] = genGradient(f,dx)
    else:
        df_dx = -g; df_dy = f

    [d2f_dxdx,d2f_dxdy,d2f_dxdz] = genGradient(df_dx,dx)
    [d2f_dydx,d2f_dydy,d2f_dydz] = genGradient(df_dy,dx)
    
    Hess = [d2f_dxdx, d2f_dxdy, d2f_dxdy, d2f_dydy]
    return np.array(Hess)

#Use the Hessian to find saddle (x) and min/max (o) points
def getXOPoints(f, critPoints, dx=[1.,1.], g=None):
    Hess = getHessian(f, g, dx)
    numP = np.shape(critPoints)[1]

    xpts = []; optsMax = []; optsMin = []
    for ip in range(numP):
        ix = critPoints[0][ip];  iy = critPoints[1][ip]
        evals = np.linalg.eigvals( np.reshape(Hess[:,ix,iy], (2,2))  )
        prod = np.prod(evals)

        if prod == 0.:
            print('Degenerate critical point found!')
        elif prod < 0.:
            xpts.append([ix,iy])
        elif evals[0] > 0.:
            optsMin.append([ix,iy])
        else:
            optsMax.append([ix,iy])

    print('X Points found ', len(xpts))
    print('O Points found ', len(optsMax) + len(optsMin))

    xpts = np.reshape(xpts, (len(xpts), 2) )
    optsMax = np.reshape(optsMax, (len(optsMax), 2) )
    optsMin = np.reshape(optsMin, (len(optsMin), 2) )


    return xpts, optsMax, optsMin

