import numpy as np


def initPolar(k, bgDir):

    dims = np.shape(k)[0]
    
    if dims == 1: 
        akp = [] 
        nbin = 0
        polar_index = []
        akplim = []
    else:
        #Setup bins and wavenumbers for perpendicular polar spectra
        perpind = np.squeeze(np.where(np.arange(dims) != bgDir))
        nkp = np.array([len(k[d]) for d in range(len(perpind))])
        kperp = [k[perpind[d]] for d in range(len(perpind))]
        nkpolar = int(np.floor(np.sqrt( np.sum(((nkp)/2)**2) )))
        nkx = nkp[0]
        nky = nkp[1]
        
        nbin = np.zeros(nkpolar) #Number of kx,ky in each polar bins
        polar_index = np.zeros((nkx, nky), dtype=int) #Polar index to simplify binning 
        if nkx == 1 & nky==1:
            dkp = 0
        elif nkx == 1:
            dkp = kperp[1]
        elif nky == 1:
            dkp = kperp[0]
        else:
            dkp = max(kperp[0][1], kperp[1][1])
        akp = (np.linspace(1, nkpolar, nkpolar))*dkp #Kperp grid
        akplim = dkp/2 + (np.linspace(0,nkpolar, nkpolar+1))*dkp #Bin limits
        #Re-written to avoid loops. Necessary for large grids.
        [kxg, kyg] = np.meshgrid(kperp[1],kperp[0]) #Deal with meshgrid weirdness (so do not have to transpose)
        kp = np.sqrt(kxg**2 + kyg**2)
        pn  = np.where(kp >= akplim[nkpolar])
        polar_index[pn[0], pn[1]] = nkpolar-1  
        nbin[nkpolar-1] = nbin[nkpolar-1] + len(pn[0])
        for ik in range(0, nkpolar):
            pn = np.where((kp < akplim[ik+1]) & (kp >= akplim[ik]))
            polar_index[pn[0], pn[1]] = ik
            nbin[ik] = nbin[ik] + len(pn[0])

        nbinDiv = nbin.copy()
        nbinDiv[np.where(nbinDiv==0)]=1 #Deal with divide by zero in ebinCorr
        ebinCorr = np.pi*akp/(akp[0]*nbinDiv)
        
    return akp, nbin, polar_index, akplim, ebinCorr


def polarFFTBin(k, bgDir, polar_index, fft_matrix):
    
    dims = np.shape(k)[0]
    perpind = np.squeeze(np.where(np.arange(dims) != bgDir))
    nkp = np.array([len(k[d]) for d in range(len(perpind))])
    nkpolar = int(np.floor(np.sqrt( np.sum(((nkp)/2)**2) )))

    if dims == 1:
        fft_isok = [] 
    elif dims == 2:
        fft_isok = np.zeros(nkpolar)
    
        for i in range(0, nkp[0]):
            for j in range(0, nkp[1]):
                if not (i == 0 and j == 0):
                    ikperp = polar_index[i,j]   
                    fft_isok[ikperp] = fft_isok[ikperp] + np.abs(fft_matrix[i,j])**2
    else:
        nkb = bgDir
        fft_isok = np.zeros(nkpolar, nkb)

        newFFTMatrix = np.transpose(fft_matrix, (perpind[0], perpind[1], bgDir)) #Re-order matrix into [perp1, perp2, bgDir]
        
        for i in range(0, nkp[0]):
            for j in range(0, nkp[1]):
                for k in range(0, nkb):
                    ikperp = polar_index[i,j]
                    fft_isok[ikperp,k] = fft_isok[ikperp,k] + np.abs(newFFTMatrix[i,j,k])**2

    return fft_isok
