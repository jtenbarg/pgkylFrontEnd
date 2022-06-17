import numpy as np

def getGradient(var, dx):

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
    

