import numpy as np
import postgkyl as pg

def getData(fileName):
    komega = pg.GData(fileName).get_values()
    k = [komega[:,d] for d in range(3)]
  

    numk = len(k[0][:])
    ntot = komega.shape[1]
    numEig = int((ntot-3)/2)

    # collect all real-part of eigenvalues
    omega_r = np.zeros((numk, numEig), float)
    omega_i = np.zeros((numk, numEig), float)
    for i in range(numk):
        omega_r[i,:] = komega[i,3::2]
        omega_r.sort()
        omega_i[i,:] = komega[i,4::2]
        omega_i.sort()
        
    omega = np.zeros((numk, numEig,2), float)
    omega[...,0] = omega_r
    omega[...,1] = omega_i
    return k, omega
