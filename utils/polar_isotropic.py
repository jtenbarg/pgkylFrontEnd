#!/usr/bin/env python
"""
Postgkyl module for binning isotropic data
"""
import numpy as np

def polar_isotropic(nkpolar, nkx, nky, nkz, polar_index, fft_matrix, ebinCorr):
    #if 2D, then nkz = kz = 0
    
    fft_isok = np.zeros(nkpolar)
    if nkz == 0:
        for i in range(0, nkx):
            for j in range(0, nky):
                ikperp = polar_index[i,j]
                fft_isok[ikperp] = fft_isok[ikperp] + 0.5*np.abs(fft_matrix[i,j])**2*ebinCorr[ikperp]
    else:
        for i in range(0, nkx):
            for j in range(0, nky):
                for k in range(0, nkz):
                    ikperp = polar_index[i,j,k]
                    fft_isok[ikperp] = fft_isok[ikperp] + 0.5*np.abs(fft_matrix[i,j,k])**2*ebinCorr[ikperp]

    #fft_isok = fft_isok/nbin[:]
    return fft_isok

