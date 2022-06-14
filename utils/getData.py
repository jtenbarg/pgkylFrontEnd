import numpy as np
import scipy as sp
import postgkyl as pg
from utils import getSlice
from pathlib import Path

def getData(self):
    
    varidGlobal = self.varid
           
    fieldvars = ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'potE', 'potB']
    dvars = ['n']
  
    if self.dimsV==0: #For fluid data
        if self.model == '5m':
            momvars = ['n', 'ux', 'uy', 'uz','pxx']
        elif self.model == '10m':
            momvars = ['n', 'ux', 'uy', 'uz','pxx', 'pxy', 'pxz', 'pyy', 'pyz', 'pzz']
    elif self.dimsV==1:
        momvars = ['ux']
        fieldvars = ['ex','potE','potB']
        pvars = ['pxx']
        qvars = ['qxxx']
    elif self.dimsV==2:
        momvars = ['ux', 'uy']
        pvars = ['pxx', 'pxy', 'pyy']
        qvars = ['qxxx','qxxy','qxyy','qyyy']
    else:
        momvars = ['ux', 'uy', 'uz']
        pvars = ['pxx', 'pxy', 'pxz', 'pyy', 'pyz', 'pzz']
        qvars = ['qxxx','qxxy','qxxz','qxyy','qxyz','qxzz','qyyy','qyyz','qyzz','qzzz']

    dof = 0
    if self.po > 0:
        nmax = int(min(self.dimsX, np.floor(self.po/2)))
        for i in range(nmax+1):
            dof = dof + int(2**(self.dimsX-i)*sp.special.comb(self.dimsX,i)*sp.special.comb(self.po-i,i))
            
    def genRead(filename,index):
        zs = getSlice.preSlice(self,filename)
        if self.model == 'vm':
            comp = str(index*dof) + ':' + str((index+1)*dof)
            data0 = pg.GData(filename, comp=comp, z0=zs[0], z1=zs[1], z2=zs[2], z3=zs[3], z4=zs[4], z5=zs[5])
            if varidGlobal[0:4] == 'dist':
                data0 = pg.GData(filename, z0=zs[0], z1=zs[1], z2=zs[2], z3=zs[3], z4=zs[4], z5=zs[5])
            polyOrder = self.po
            #if not (isinstance(self.params.get('polyOrderOverride'), type(None))):
            #    polyOrder = self.params["polyOrderOverride"]
            
            proj = pg.GInterpModal(data0, polyOrder, self.basis)
            coords, data = proj.interpolate()
        elif self.model == '5m' or self.model == '10m':
            comp = index
            data0 = pg.data.GData(filename, comp=comp, z0=zs[0], z1=zs[1], z2=zs[2])
            data = data0.getValues()
            coords = data0.getGrid()
            if self.suffix == '.gkyl':
                data = data[...,comp]
        else:
            raise RuntimeError("You have confused me! I don't know what to do with data of type {0}.".format(self.model))
        self.time = data0.meta['time']
        
        return coords, data

    def getDist(varid): #Returns particle distribution
        spec = varid[varid.find('_')+1:] + '_'
        filename = self.filenameBase + spec + str(self.fileNum) + self.suffix
        return genRead(filename,0)

    def getGenField(varid): #Returns E* or B*
        index = fieldvars.index(varid[0:2])
        filename = self.filenameBase + 'field_' + str(self.fileNum) + self.suffix
        
        return genRead(filename, index)

    def getDens(varid): #Return M0 = n
        spec = varid[varid.find('_')+1:]
        if self.model == 'vm':
            index = dvars.index(varid[0])
            filename = self.filenameBase + spec + '_M0_' + str(self.fileNum) + self.suffix
            coords, data = genRead(filename, index)
        elif self.model == '5m' or self.model == '10m':
            index = momvars.index(varid[0])
            specIndex = self.speciesFileIndex.index(spec)
            filename = self.filenameBase + spec + '_' + str(self.fileNum) + self.suffix
            coords, data = genRead(filename, index)
            data = data / self.mu[specIndex]
        return coords, data

    def getGenMom(varid): #Returns M1 = n*u
        try:
            index = momvars.index(varid[0:2])
            spec = varid[varid.find('_')+1:]
            if self.model == 'vm':
                filename = self.filenameBase + spec + '_M1i_' + str(self.fileNum) + self.suffix
                coords, data = genRead(filename, index)
            elif self.model == '5m' or self.model == '10m':
                filename = self.filenameBase + spec + '_' + str(self.fileNum) + self.suffix
                coords, data = genRead(filename, index)
                specIndex = self.speciesFileIndex.index(spec)
                data = data / self.mu[specIndex]
            return coords, data
        except:
            print('Moment {0} not found.'.format(varid[0:2]))
            return [[0.]], np.array([0.])
        

    def getU(varid):#Return u = M1 / M0
        spec = varid[varid.find('_')+1:]
        coords, m1 = getGenMom(varid)
        coords, m0 = getDens('n_' + spec)
        return coords, m1 / m0

    def getGenP(varid): #Return M2
        try:
            spec = varid[varid.find('_')+1:]
            if self.model == 'vm':
                index = pvars.index(varid[0:3])
                filename = self.filenameBase + spec + '_M2ij_' + str(self.fileNum) + self.suffix
                if not Path(filename).is_file():
                    print('Warning: Full moment file does not exist. Using M2 instead.')
                    filename = self.filenameBase + spec + '_M2_' + str(self.fileNum) + self.suffix
                coords, data = genRead(filename, index)
            elif self.model == '5m' or self.model == '10m':
                filename = self.filenameBase + spec + '_' + str(self.fileNum) + self.suffix
                index = momvars.index(varid[0:3])
                coords, data = genRead(filename, index)
                specIndex = self.speciesFileIndex.index(spec)
                data = data / self.mu[specIndex]
                
            return coords, data
        except:
            print('Moment {0} not found.'.format(varid[0:3]))
            return [[0.]], np.array([0.])

    def getGenQ(varid): #Return M3
        index = pvars.index(varid[0:3])
        spec = varid[varid.find('_')+1:]
        filename = self.filenameBase + spec + '_M3ijk_' + str(self.fileNum) + self.suffix
        if not Path(filename).is_file():
            print('Warning: Full moment file does not exist. Using M3i instead.')
            filename = self.filenameBase + spec + '_M3i_' + str(self.fileNum) + self.suffix
            if not Path(filename).is_file():
                print('Warning: No heat flux data to read.')
                coords = 0.
                data = 0.
            else:
               coords, data = genRead(filename, index)
       
        coords, data = genRead(filename, index)
        return coords, data

    def getMagB(varid): #Return |B|
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        data = np.sqrt(bx**2 + by**2 + bz**2)
        return coords, data

    def getMagE(varid): #Return |E|
        coords, ex = getGenField('ex')
        coords, ey = getGenField('ey')
        coords, ez = getGenField('ez')
        data = np.sqrt(ex**2 + ey**2 + ez**2)
        return coords, data

    def getPsi(varid): #Return psi
        if self.dimsX < 2:
            coords = 0.
            psi = 0.
        else:
            coords, bx = getGenField('bx')
            coords, by = getGenField('by')
            dx = np.zeros(len(coords))
            for d in range(len(coords)):
                dx[d] = coords[d][1] - coords[d][0]
        
            psi = np.zeros(bx.shape)
            for ix in range(1, bx.shape[0]):
                psi[ix, 0] = psi[ix-1,0] - by[ix,0]*dx[0]
            for iy in range(1,  bx.shape[1]):
                psi[0:, iy] = psi[0:, iy-1] + bx[0:, iy]*dx[1]
        return coords, psi

    def getStream(varid): #Return stream function
        if self.dimsX < 2:
            coords = 0.
            stream = 0.
        else:
            spec = varid[varid.find('_')+1:]
            coords, uy = getGenMom('uy' + '_' + spec)
            coords, ux = getGenMom('ux' + '_' + spec)
            
            dx = np.zeros(len(coords))
            for d in range(len(coords)):
                dx[d] = coords[d][1] - coords[d][0]
        
            stream = np.zeros(ux.shape)
            for ix in range(1, ux.shape[0]):
                stream[ix, 0] = stream[ix-1,0] - uy[ix,0]*dx[0]
            for iy in range(1,  ux.shape[1]):
                stream[0:, iy] = stream[0:, iy-1] + ux[0:, iy]*dx[1]
        return coords, stream

    def getPhi(varid): #Return phi
        if self.dimsX < 2:
            coords = 0.
            phi = 0.
        else:
            coords, ex = getGenField('ex')
            coords, ey = getGenField('ey')
            dx = np.zeros(len(coords))
            for d in range(len(coords)):
                dx[d] = coords[d][1] - coords[d][0]
        
            phi = np.zeros(ex.shape)
            for ix in range(1, ex.shape[0]):
                phi[ix, 0] = phi[ix-1,0] + ex[ix,0]*dx[0]
            for iy in range(1,  ex.shape[1]):
                phi[0:, iy] = phi[0:, iy-1] + ey[0:, iy]*dx[1]
        return coords, -phi
   
    
    def getEpar(varid): #Return E.B
        coords, ex = getGenField('ex')
        coords, ey = getGenField('ey')
        coords, ez = getGenField('ez')
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        B = np.sqrt(bx**2 + by**2 + bz**2)
        data = (ex*bx + ey*by + ez*bz) / B
        return coords, data

    def getJ(varid): #Return various forms of the current
        nspec = len(self.mu)
        specInd = varid.find('_')+1
        if specInd:
            spec = varid[varid.find('_')+1:]
            specIndex = self.speciesFileIndex.index(spec)
            coords, tmp = getGenMom('u' + varid[1] + '_' + spec)
            data =  self.q[specIndex]*tmp
        else:
            for i in range(nspec):
                spec = self.speciesFileIndex[i]
                specIndex = self.speciesFileIndex.index(spec)
                coords, tmp = getGenMom('u' + varid[1] + '_' + spec)
                if i == 0:
                    data = self.q[i]*tmp
                else:
                    data = data + self.q[i]*tmp
        return coords, data

    def getJPar(varid): #Return J.B
        suf = ['x', 'y', 'z']
        specInd = varid.find('_')+1
        if specInd:
            spec = '_' + varid[varid.find('_')+1:]
            suf = [s + spec for s in suf]
        for id in range(self.dimsV):
            coords, b = getGenField('b' + suf[id])
            coords, j = getJ('j' + suf[id])
            if id == 0:
                data = j*b
                b2 = b*b
            else:
                data = data + j*b
                b2 = b2 + b*b
            
        return coords, data / np.sqrt(b2)

    def getWork(varid): #Return various combos of J.E
        suf = ['x', 'y', 'z']
        specInd = varid.find('_')+1
        
        if specInd:
            spec = '_' + varid[varid.find('_')+1:]
            suf = [s + spec for s in suf]
        
        if varid.find('par') != -1: #Parallel work
            coords, jpar = getJPar(varid)
            coords, epar = getEpar(varid)
            data = jpar*epar
        elif any(varid.find(elem) != -1 for elem in suf): #Component work
            for elem in suf:
                if varid.find(elem) != -1:
                    comp = elem
            coords, e = getGenField('e' + comp)
            coords, j = getJ('j' + comp)
            data = j*e
        else: #Full J.E
            for id in range(self.dimsV):
                coords, e = getGenField('e' + suf[id])
                coords, j = getJ('j' + suf[id])
                if id == 0:
                    data = j*e
                else:
                    data = data + j*e

        return coords, data
    
    def getPress(varid): #Return Pij component, optionally in restframe
        coords, data = getGenP(varid)
        spec = varid[varid.find('_')+1:]
      
        if self.params["restFrame"]:
            coords, nu = getGenMom('u' + varid[1] + '_' + spec)
            coords, nv = getGenMom('u' + varid[2] + '_' + spec)
            coords, n = getDens('n' + '_' + spec)
            data = data - (nu*nv) / n
        specIndex = self.speciesFileIndex.index(spec)
        data = data*self.mu[specIndex]
        return coords, data
      
    def getTrP(varid): # Return Tr(P)
        spec = '_' + varid[varid.find('_')+1:]
        coords, pyy = getPress('pyy' + spec)
        coords, pzz = getPress('pzz' + spec)
        coords, pxx = getPress('pxx' + spec)
        data = pxx + pyy + pzz
        return coords, data

    def getPressPar(varid): #Return ppar = Pij bi bj
        spec = '_' + varid[varid.find('_')+1:]
        coords, pxx = getPress('pxx' + spec)
        coords, pyy = getPress('pyy' + spec)
        coords, pzz = getPress('pzz' + spec)
        coords, pxy = getPress('pxy' + spec)
        coords, pxz = getPress('pxz' + spec)
        coords, pyz = getPress('pyz' + spec)
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        B = np.sqrt(bx**2 + by**2 + bz**2)
        bx = bx / B
        by = by / B
        bz = bz / B

        data = pxx*bx**2 + pyy*by**2 + pzz*bz**2 + 2.*(pxy*bx*by + pxz*bx*bz + pyz*by*bz)
        
        return coords, data

    def getPressPerp(varid): #Return Pperp = (Tr(P) - Ppar) / 2
        spec = '_' + varid[varid.find('_')+1:]
        coords, trp = getTrP(spec)
        coords, ppar = getPressPar(varid)

        data = (trp - ppar) / 2.
        return coords, data

    def getTemp(varid): #Return T = P / n
        spec = '_' + varid[varid.find('_')+1:]
        coords, trp = getTrP(spec)
        coords, n = getDens('n' + spec)
        data = (trp / n) / self.dimsV
        return coords, data

    def getTempPar(varid): #Return Tpar = Ppar / n
        spec = '_' + varid[varid.find('_')+1:]
        coords, ppar = getPressPar(spec)
        coords, n = getDens('n' + spec)
        data = ppar / n 
        return coords, data

    def getTempPerp(varid): #Return Tperp = Pperp / n
        spec = '_' + varid[varid.find('_')+1:]
        coords, pperp = getPressPerp(spec)
        coords, n = getDens('n' + spec)
        data = pperp / n
        return coords, data

    def getTParPerp(varid): #Return Tpar / Tperp
        spec = '_' + varid[varid.find('_')+1:]
        coords, ppar = getPressPar(spec)
        coords, pperp = getPressPerp(spec)            
        return coords, ppar / pperp

    def getTPerpPar(varid): #Return Tperp / Tpar
        spec = '_' + varid[varid.find('_')+1:]
        coords, ppar = getPressPar(spec)
        coords, pperp = getPressPerp(spec)            
        return coords, pperp / ppar

    def getAgyro(varid): #Return Swisdak 2015 agyrotropy measure
        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame

        spec = '_' + varid[varid.find('_')+1:]
        coords, pxx = getPress('pxx' + spec)
        coords, pyy = getPress('pyy' + spec)
        coords, pzz = getPress('pzz' + spec)
        coords, pxy = getPress('pxy' + spec)
        coords, pxz = getPress('pxz' + spec)
        coords, pyz = getPress('pyz' + spec)
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        B = np.sqrt(bx**2 + by**2 + bz**2)
        bx = bx / B
        by = by / B
        bz = bz / B

        ppar = pxx*bx**2 + pyy*by**2 + pzz*bz**2 + 2.*(pxy*bx*by + pxz*bx*bz + pyz*by*bz)
        I1 = pxx + pyy + pzz
        I2 = pxx*pyy + pxx*pzz + pyy*pzz - (pxy**2 + pxz**2 + pyz**2)
        denom = (I1 - ppar)* (I1 + 3.*ppar) 
        data = np.sqrt( np.absolute( 1. - 4.* I2/denom))
        self.params["restFrame"] = tmp
        return coords, data

    def getBeta(varid):
        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame
        coords, B = getMagB(varid)
        spec = varid[varid.find('_')+1:]
        if varid.find('par') > -1:
            coords, p = getPressPar(varid)
        elif varid.find('perp') > -1:
            coords, p = getPressPerp(varid)
        else:
            coords, p = getTrP(varid)
            p = p / self.dimsV
        self.params["restFrame"] = tmp
        data = 2*p*self.mu0 / B**2
        return coords, data

    def getMu(varid):
        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame
        coords, B = getMagB(varid)
        spec = varid[varid.find('_')+1:]
        coords, p_perp = getPressPerp(varid)
        coords, n = getDens('n_' + spec)
        self.params["restFrame"] = tmp
        return coords, p_perp / (B*n)

    def getGyroradius(varid):
        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame
        coords, B = getMagB(varid)
        spec = varid[varid.find('_')+1:]
        coords, T_perp = getTempPerp(varid)
        specIndex = self.speciesFileIndex.index(spec)
        vt = np.sqrt(2.*T_perp / self.mu[specIndex])
        Omega = np.abs(self.q[specIndex])*B / self.mu[specIndex]
        self.params["restFrame"] = tmp
        return coords, vt/Omega

    def getInertialLength(varid):
        spec = varid[varid.find('_')+1:]
        specIndex = self.speciesFileIndex.index(spec)
        c = 1 / np.sqrt(self.eps0*self.mu0)
        coords, n = getDens('n_' + spec)
        omegaP = np.sqrt(np.abs(n) * self.q[specIndex]**2. / (self.mu[specIndex]*self.eps0))
        return coords, c / omegaP

    def getDebyeLength(varid):
        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame
        spec = varid[varid.find('_')+1:]
        specIndex = self.speciesFileIndex.index(spec)
        coords, T = getTemp(varid)
        coords, n = getDens('n_' + spec)
        debye = np.sqrt(T*self.eps0 / (np.abs(n) * self.q[specIndex]**2.))
        self.params["restFrame"] = tmp
        return coords, debye
        

        
    ####################################################################
    #End read functions
    #################################################################### 
    
    if varidGlobal[0:4] == 'dist':
        coords, data = getDist(varidGlobal)
    elif varidGlobal[0:4] == 'beta':
        coords, data = getBeta(varidGlobal)
    elif varidGlobal[0] == 'b': #magnetic fields
        coords, data = getGenField(varidGlobal)
    elif varidGlobal[0] == 'e':
        if varidGlobal[1:4] == 'par':
            coords, data = getEpar(varidGlobal)
        else:
            coords, data = getGenField(varidGlobal)
    elif varidGlobal[0] == 'n': #density
        coords, data = getDens(varidGlobal)
    elif varidGlobal[0] == 'u': #flow velocity
        coords, data = getU(varidGlobal)
    elif varidGlobal[0:3] == 'psi': #2D flux function
        coords, data = getPsi(varidGlobal)
    elif varidGlobal[0:3] == 'phi': #2D potential function
        coords, data = getPhi(varidGlobal)
    elif varidGlobal[0:6] == 'stream': #2D stream function
        coords, data = getStream(varidGlobal)
    elif varidGlobal[0] == 'p': #pressure
        if varidGlobal[1:4] == 'par': #Parallel pressure
            coords, data = getPressPar(varidGlobal)
        elif varidGlobal[1:5] == 'perp': #Perp Pressure
            coords, data = getPressPerp(varidGlobal)
        else:# Pressure component
            coords, data = getPress(varidGlobal)
    elif varidGlobal[0:3] == 'trp': #Tr(P)
        coords, data = getTrP(varidGlobal)
    elif varidGlobal[0:4] == 'temp': #Temperature
        if varidGlobal[4:7] == 'par':
            if varidGlobal[7:11] == 'perp': #Tpar / Tperp
                coords, data = getTParPerp(varidGlobal)
            else: 
                coords, data = getTempPar(varidGlobal)  #Parallel temperature
        elif varidGlobal[4:8] == 'perp': 
            if varidGlobal[8:11] == 'par': #Tperp / Tpar
                coords, data = getTPerpPar(varidGlobal)
            else:
                coords, data = getTempPerp(varidGlobal) #Perpendicular temperature
        else:
            coords, data = getTemp(varidGlobal)
    elif varidGlobal[0:3] == 'mag':
        if varidGlobal[3] == 'e': #|E|
            coords, data = getMagE(varidGlobal)
        elif varidGlobal[3] == 'b': #|B|
            coords, data = getMagB(varidGlobal)
    elif varidGlobal[0:5] == 'agyro': #Swisdak 2015 agyrotropy
        coords, data = getAgyro(varidGlobal)
    elif varidGlobal[0] == 'j': #Various current definitions
        if varidGlobal[1:4] == 'par':
            coords, data = getJPar(varidGlobal)
        else:
            coords, data = getJ(varidGlobal)
    elif varidGlobal[0:4] == 'work': #Various forms of J.E work
        coords, data = getWork(varidGlobal)
    elif varidGlobal[0:2] == 'mu':
        coords, data = getMu(varidGlobal)
    elif varidGlobal[0:3] == 'rho':
        coords, data = getGyroradius(varidGlobal)
    elif varidGlobal[0:8] == 'inertial':
        coords, data = getInertialLength(varidGlobal)
    elif varidGlobal[0:5] == 'debye':
        coords, data = getDebyeLength(varidGlobal)
    else:
        coords = [[0.]]
        data = [0.]
        print('Unrecognized variable name {0}! You have confused me, so no data for you!'.format(varidGlobal))

    
    dims = len(np.shape(data)) - 1
    # Center the grid values.
    self.dx = np.zeros(dims)
    if dims > 0:
        for d in range(dims):
            self.dx[d] = coords[d][1] - coords[d][0]
            coords[d] = 0.5*(coords[d][:-1] + coords[d][1:])
  
    self.coords = coords
    #if dims == 1:
    #    self.coords = coords[0] #Simplifies some later operations to bypass the list format
    self.data = data

    getSlice.postSlice(self)
   
