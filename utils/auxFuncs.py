import numpy as np

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
