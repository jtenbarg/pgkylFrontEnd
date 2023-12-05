import numpy as np
import scipy as sp
import postgkyl as pg
from utils import getSlice
from utils import auxFuncs
try:
    import adios 
except ModuleNotFoundError:
    adios2 = 1


from pathlib import Path
def getData(self):
    
    varidGlobal = self.varid
    if self.model == 'vp':
        fieldvars = ['ph', 'ax', 'ay', 'az']
    else:
        fieldvars = ['ex', 'ey', 'ez', 'bx', 'by', 'bz', 'potE', 'potB']
    
    dvars = ['n']
    tracePVars = 0  
    traceQVars = 0 #For handling M2 and M3i data correctly
    if self.dimsV==0: #For fluid data
        if self.model == '5m':
            momvars = ['n', 'ux', 'uy', 'uz','p']
        elif self.model == '10m':
            momvars = ['n', 'ux', 'uy', 'uz','pxx', 'pxy', 'pxz', 'pyy', 'pyz', 'pzz']
    #elif self.model == 'pkpm':
    #   moments = ['n','ppar','pperp','qpar','qperp','rparperp','rperpperp']
    #   momvars = ['ux','uy','uz']
    elif self.model == 'pkpm':
        momvars = ['n', 'ux', 'uy', 'uz','pxx', 'pxy', 'pxz', 'pyy', 'pyz', 'pzz']
        moments = ['n','M1','ppar','pperp','qpar','qperp','rparperp','rperpperp']
        auxvars = ['ux','uy','uz','Tperp/m','m/Tperp','1/rho div(p_parallel b_hat)', '1/rho p_perp div(b)', 'bb : grad(u)']
    elif self.dimsV==1:
        momvars = ['ux']
        pvars = ['pxx']
        qvarsijk = ['qxxx']; qvarsi = ['qx']
    elif self.dimsV==2:
        momvars = ['ux', 'uy']
        pvars = ['pxx', 'pxy', 'pyy']
        qvarsijk = ['qxxx','qxxy','qxyy','qyyy']; qvarsi = ['qx','qy']
        
    else:
        momvars = ['ux', 'uy', 'uz']
        pvars = ['pxx', 'pxy', 'pxz', 'pyy', 'pyz', 'pzz']
        qvarsijk = ['qxxx','qxxy','qxxz','qxyy','qxyz','qxzz','qyyy','qyyz','qyzz','qzzz']
        qvarsi = ['qx','qy', 'qz']
    dof = 0
    if self.po > 0:
        nmax = int(min(self.dimsX, np.floor(self.po/2)))
        for i in range(nmax+1):
            dof = dof + int(2**(self.dimsX-i)*sp.special.comb(self.dimsX,i)*sp.special.comb(self.po-i,i))
    
    def genRead(filename,index):
        zs = getSlice.preSlice(self,filename)
        if self.model == 'vm' or self.model == 'pkpm' or self.model == 'vp':
            comp = str(index*dof) + ':' + str((index+1)*dof)
            polyOrder = self.po
            basis = self.basis
            data0 = pg.GData(filename, comp=comp, z0=zs[0], z1=zs[1], z2=zs[2], z3=zs[3], z4=zs[4], z5=zs[5])
            if varidGlobal[0:4] == 'dist':
                data0 = pg.GData(filename, z0=zs[0], z1=zs[1], z2=zs[2], z3=zs[3], z4=zs[4], z5=zs[5])
                if self.model == 'pkpm':
                    if polyOrder == 2:
                        basis = 'ms'
                    else:
                        basis = 'pkpmhyb'
                    
            
            if not (isinstance(self.params.get('polyOrderOverride'), type(None))):
                proj = pg.GInterpModal(data0, polyOrder, basis, self.params["polyOrderOverride"])
            else:
                proj = pg.GInterpModal(data0, polyOrder, basis)
            if self.suffix == '.gkyl':
                coords, data = proj.interpolate(index)
            else:
                coords, data = proj.interpolate()
        elif self.model == '5m' or self.model == '10m':
            comp = index
            data0 = pg.data.GData(filename, comp=comp, z0=zs[0], z1=zs[1], z2=zs[2])
            data = data0.getValues()
            coords = data0.getGrid()
            if self.suffix == '.gkyl':
                data = data[...,comp]
                data = data[...,np.newaxis]

        else:
            raise RuntimeError("You have confused me! I don't know what to do with data of type {0}.".format(self.model))
        if self.suffix == '.gkyl':
            print('Warning, gkyl0 data files do not contain time date. Time set to fileNum')
            self.time = self.fileNum
        else:
            if adios2:
                self.time = data0.ctx['time']
            else:
                self.time = data0.meta['time'] 
            if self.time is None:
                print('Warning, data file does not contain time date. Time set to fileNum')
                self.time = self.fileNum
        return coords, data

    def getDist(varid): #Returns particle distribution
        spec = varid[varid.find('_')+1:] + '_'
        filename = self.filenameBase + spec + str(self.fileNum) + self.suffix
        index = 0
        if self.model == 'pkpm':
            index = int(varid[4])
        return genRead(filename,index)

    def getGenField(varid): #Returns E* or B*
        index = fieldvars.index(varid[0:2])
        filename = self.filenameBase + 'field_' + str(self.fileNum) + self.suffix
        
        return genRead(filename, index)

    def getDens(varid): #Return M0 = n
        spec = varid[varid.find('_')+1:]
        if self.model == 'vm' or self.model == 'vp':
            index = dvars.index(varid[0])
            filename = self.filenameBase + spec + '_M0_' + str(self.fileNum) + self.suffix
            coords, data = genRead(filename, index)
        elif self.model == '5m' or self.model == '10m':
            index = momvars.index(varid[0])
            specIndex = self.speciesFileIndex.index(spec)
            filename = self.filenameBase + spec + '_' + str(self.fileNum) + self.suffix
            coords, data = genRead(filename, index)
            data = data / self.mu[specIndex]
        elif self.model == 'pkpm':
            index = moments.index(varid[0])
            specIndex = self.speciesFileIndex.index(spec)
            filename = self.filenameBase + spec + '_pkpm_moms_' + str(self.fileNum) + self.suffix
            coords, data = genRead(filename, index)
            data = data / self.mu[specIndex]
        return coords, data

    def getGenMom(varid): #Returns M1 = n*u
        try:
            index = momvars.index(varid[0:2])
            spec = varid[varid.find('_')+1:]
            if self.model == 'vm' or self.model == 'vp':
                filename = self.filenameBase + spec + '_M1i_' + str(self.fileNum) + self.suffix
                coords, data = genRead(filename, index)
            elif self.model == '5m' or self.model == '10m':
                filename = self.filenameBase + spec + '_' + str(self.fileNum) + self.suffix
                coords, data = genRead(filename, index)
                specIndex = self.speciesFileIndex.index(spec)
                data = data / self.mu[specIndex]
            elif self.model == 'pkpm':
                filename = self.filenameBase + spec + '_pkpm_fluid_' + str(self.fileNum) + self.suffix
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
        nonlocal tracePVars
        try:
            spec = varid[varid.find('_')+1:]
            if self.model == 'vm' or self.model == 'vp':
                index = pvars.index(varid[0:3])
                filename = self.filenameBase + spec + '_M2ij_' + str(self.fileNum) + self.suffix
                if not Path(filename).is_file():
                    print('Warning: Full moment file does not exist. Using M2 instead.')
                    tracePVars = 1
                    filename = self.filenameBase + spec + '_M2_' + str(self.fileNum) + self.suffix
                coords, data = genRead(filename, index)
            elif self.model == '5m' or self.model == '10m':
                filename = self.filenameBase + spec + '_' + str(self.fileNum) + self.suffix
                index = momvars.index(varid[0:3])
                coords, data = genRead(filename, index)
                specIndex = self.speciesFileIndex.index(spec)
                data = data / self.mu[specIndex]
            elif self.model == 'pkpm':
                if varid[1] == 'p': #ppar or pperp
                    filename = self.filenameBase + spec + '_pkpm_moms_' + str(self.fileNum) + self.suffix
                    index = moments.index(varid[0:varid.find('_')])
                else:
                    filename = self.filenameBase + spec + '_pkpm_fluid_' + str(self.fileNum) + self.suffix
                    index = momvars.index(varid[0:varid.find('_')])
                coords, data = genRead(filename, index)
                specIndex = self.speciesFileIndex.index(spec)
                data = data / self.mu[specIndex]
                
            return coords, data
        except:
            print('Moment {0} not found.'.format(varid[0:3]))
            return [[0.]], np.array([0.])

    def getGenQ(varid): #Return M3
        nonlocal traceQVars
        try:
            spec = varid[varid.find('_')+1:]
            if self.model == 'vm' or self.model == 'vp':     
                filename = self.filenameBase + spec + '_M3ijk_' + str(self.fileNum) + self.suffix
                if Path(filename).is_file():
                    index = qvarsijk.index(varid[0:4])
                if not Path(filename).is_file():
                    print('Warning: Full moment file does not exist. Using M3i instead.')
                    traceQVars = 1
                    filename = self.filenameBase + spec + '_M3i_' + str(self.fileNum) + self.suffix
                    index = qvarsi.index(varid[0:2])
                if not Path(filename).is_file():
                    print('Warning: No heat flux data to read.')
                    return [[0.]], np.array([0.])
                else:
                    return genRead(filename, index)
            elif self.model == 'pkpm':
                filename = self.filenameBase + spec + '_pkpm_moms_' + str(self.fileNum) + self.suffix
                index = moments.index(varid[0:varid.find('_')])
                coords, data = genRead(filename, index)
                specIndex = self.speciesFileIndex.index(spec)
                data = data / self.mu[specIndex]
                return coords, data
        except:
            print('Moment {0} not found.'.format(varid[0:4]))
            return [[0.]], np.array([0.])
              
       

    def getMagB(varid): #Return |B|
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        data = np.sqrt(bx**2 + by**2 + bz**2)
        return coords, data

    def getDivB(varid): #Return div B
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        dims = len(np.shape(bx)) - 1
        dx = np.zeros(dims)
        for d in range(dims):
            dx[d] = coords[d][1] - coords[d][0]
        [duxdx,duxdy,duxdz] = auxFuncs.genGradient(bx,dx)
        [duydx,duydy,duydz] = auxFuncs.genGradient(by,dx)
        [duzdx,duzdy,duzdz] = auxFuncs.genGradient(bz,dx)

        div = (duxdx + duydy + duzdz)
        
        return coords, div

    def getMagE(varid): #Return |E|
        coords, ex = getGenField('ex')
        coords, ey = getGenField('ey')
        coords, ez = getGenField('ez')
        data = np.sqrt(ex**2 + ey**2 + ez**2)
        return coords, data

    def getDivE(varid): #Return div E
        coords, ex = getGenField('ex')
        coords, ey = getGenField('ey')
        coords, ez = getGenField('ez')
        dims = len(np.shape(ex)) - 1
        dx = np.zeros(dims)
        for d in range(dims):
            dx[d] = coords[d][1] - coords[d][0]
        [duxdx,duxdy,duxdz] = auxFuncs.genGradient(ex,dx)
        [duydx,duydy,duydz] = auxFuncs.genGradient(ey,dx)
        [duzdx,duzdy,duzdz] = auxFuncs.genGradient(ez,dx)

        div = (duxdx + duydy + duzdz)
        
        return coords, div

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
            coords, uy = getU('uy' + '_' + spec)
            coords, ux = getU('ux' + '_' + spec)
            
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
   
    def getEVP(varidGlobal): #Return Ei for VP data
        coords, phi = getGenField('phi')
        dx = np.zeros(len(coords))
        for d in range(len(coords)):
            dx[d] = coords[d][1] - coords[d][0]
        E = auxFuncs.genGradient(-phi,dx)
        suf = ['x', 'y', 'z']
        return coords, E[suf.index(varidGlobal[1])]

    def getBVP(varidGlobal): #Return Bi for VP data
        coords, ax = getGenField('ax')
        coords, ay = getGenField('ay')
        coords, az = getGenField('az')
        dx = np.zeros(len(coords))
        for d in range(len(coords)):
            dx[d] = coords[d][1] - coords[d][0]
        [daxdx, daxdy, daxdz] = auxFuncs.genGradient(ax,dx)
        [daydx, daydy, daydz] = auxFuncs.genGradient(ay,dx)
        [dazdx, dazdy, dazdz] = auxFuncs.genGradient(az,dx)
        B = np.zeros(3 + np.shape(daxdx))
        print(np.shape(B))
        B[0] = daydz - dazdy
        B[1] = dazdx - daxdz
        B[2] = daxdy - daydx

        suf = ['x', 'y', 'z']

        return coords, B[suf.index(varidGlobal[1])]

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
            if self.model == 'vm' or self.model == 'vp':
                idRange = self.dimsV
            else:
                idRange = 3
            for id in range(idRange):
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
      
        if self.model == '5m': 
            data = data*2. #Convert Epsilon to P_ii
            
        if self.params["restFrame"] and not (self.model == 'pkpm' and varid[1] == 'p'):
            coords, n = getDens('n' + '_' + spec)
            if self.model == '5m':           
                coords, nux = getGenMom('ux_' + spec)
                coords, nuy = getGenMom('uy_' + spec)
                coords, nuz = getGenMom('uz_' + spec)
                data = data - (nux*nux + nuy*nuy + nuz*nuz) / n
                #data = data / 3 #Return scalar p
            elif tracePVars:                
                for id in range(self.dimsV):
                    coords, nui = getGenMom(momvars[id] + '_' + spec)
                    data = data - (nui*nui) / n    
                #data = data / self.dimsV #Return scalar p     
            else:
                coords, nu = getGenMom('u' + varid[1] + '_' + spec)
                coords, nv = getGenMom('u' + varid[2] + '_' + spec)
                data = data - (nu*nv) / n
        specIndex = self.speciesFileIndex.index(spec)
        data = data*self.mu[specIndex]
        return coords, data
    
    def getHeatFlux(varid): #Return Qijk component, optionally in restframe
        coords, data = getGenQ(varid)
        spec = varid[varid.find('_')+1:]

        if self.params["restFrame"] and not self.model == 'pkpm':
            coords, P = getPress('p' + varid[1] + varid[1] + '_' + spec)
            dims = self.dimsV
            if tracePVars and not dims==1:
                print('Warning: M2ij does not exist. Cannot compute restframe Q!')
            elif traceQVars:
                coords, n = getDens('n' + '_' + spec)
                coords, nuk = getGenMom('u' + varid[1] + '_' + spec)
                #data = data / 2.
                ii = ['x', 'y', 'z']
                self.params["restFrame"] = 0
                for id in range(dims):
                    pcomp = ''.join(sorted('p' + ii[id] + varid[1]))
                    coords, Pik = getPress(pcomp + '_' + spec)
                    coords, Pii = getPress('p' + ii[id] + ii[id] + '_' + spec)
                    coords, nui = getGenMom(momvars[id] + '_' + spec)
                    data = data + 2*(nui*nui)*nuk / (n*n) - 2*nui*Pik / n - nuk*Pii / n  
                self.params["restFrame"] = 1                 
            else:
                coords, n = getDens('n' + '_' + spec)
                coords, nui = getGenMom('u' + varid[1] + '_' + spec)
                coords, nuj = getGenMom('u' + varid[2] + '_' + spec)
                coords, nuk = getGenMom('u' + varid[3] + '_' + spec)
                self.params["restFrame"] = 0
                coords, Pij = getPress('p' + varid[1] + varid[2] + '_' + spec)
                coords, Pjk = getPress('p' + varid[2] + varid[3] + '_' + spec)
                coords, Pik = getPress('p' + varid[1] + varid[3] + '_' + spec)
                self.params["restFrame"] = 1  
                data = data - (nui * Pjk + nuj*Pik + nuk*Pij) / n + 2*nui*nuj*nuk / (n*n)

        specIndex = self.speciesFileIndex.index(spec)
        data = data*self.mu[specIndex]
        return coords, data

    def getTrP(varid): # Return Tr(P)
        spec =  varid[varid.find('_')+1:]
        nonlocal tracePVars
        if self.model == 'vm' or self.model == 'vp':
            filename = self.filenameBase + spec + '_M2ij_' + str(self.fileNum) + self.suffix
            if not Path(filename).is_file():
                tracePVars = 1
        if self.model == 'pkpm':
            coords, ppar = getPress('ppar' + '_' + spec)
            coords, pperp = getPress('pperp' + '_' + spec)
            data = ppar + 2.*pperp
        elif self.model == '5m':
            coords, data = getPress('p' + '_' + spec)     
        elif tracePVars:
            coords, data = getPress('pxx' + '_' + spec)
        else:
            ii = ['yy', 'zz']
            if self.model == '10m':
                dims = 3
            else:
                dims = self.dimsV
            coords, data = getPress('pxx' + '_' + spec)
            for id in range(dims-1):
                coords, pii = getPress('p' + ii[id] + '_' + spec)
                data = data + pii

        return coords, data

    def getPressPar(varid): #Return ppar = Pij bi bj
        spec = '_' + varid[varid.find('_')+1:]
        if self.model == 'pkpm':
            coords, data = getPress('ppar' + spec)
        else:
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
        if self.model == 'pkpm':
            coords, data = getPress('pperp' + spec)
        else:
            coords, trp = getTrP(spec)
            coords, ppar = getPressPar(varid)

            data = (trp - ppar) / 2.
        return coords, data

    def getTemp(varid): #Return T = P / n
        spec = '_' + varid[varid.find('_')+1:]
        coords, trp = getTrP(spec)
        coords, n = getDens('n' + spec)
        if self.model == '10m' or self.model == '5m' or self.model == 'pkpm':
            data = (trp / n) / 3
        else:
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
        coords, n = getDens('n' + spec)
        if varid.find('par') > -1:
            coords, T = getTempPar(varid)
        elif varid.find('perp') > -1:
            coords, T = getTempPerp(varid)
        else:
            coords, T = getTemp(varid)     
        self.params["restFrame"] = tmp
        data = 2*n*T*self.mu0 / B**2
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
        
    def getExB(varid):
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        coords, ex = getGenField('ex')
        coords, ey = getGenField('ey')
        coords, ez = getGenField('ez')
        B = np.sqrt(bx**2 + by**2 + bz**2)
        
        if varid[-1] == 'x':
            ExB = (ey*bz - by*ez)/B**2
        elif varid[-1] == 'y': 
            ExB = (ez*bx - bz*ex)/B**2
        elif varid[-1] == 'z':
            ExB = (ex*by - bx*ey)/B**2
        else:
            ExB = np.zeros(np.shape(bx))
            
        return coords, ExB

    def getPoynting(varid):
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        coords, ex = getGenField('ex')
        coords, ey = getGenField('ey')
        coords, ez = getGenField('ez')
        
        mu0 = self.mu0
        if varid[-1] == 'x':
            Poynting = (ey*bz - by*ez)
        elif varid[-1] == 'y': 
            Poynting = (ez*bx - bz*ex)
        elif varid[-1] == 'z':
            Poynting = (ex*by - bx*ey)
        else:
            Poynting = np.sqrt((ey*bz - by*ez)**2 + (ez*bx - bz*ex)**2 + (ex*by - bx*ey)**2)
            
        return coords, Poynting/mu0

    def getCrossHelicity(varid):
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        
        spec = varid[varid.find('_')+1:]
        specIndex = self.speciesFileIndex.index(spec)
        mu = self.mu[specIndex]
        mu0 = self.mu0
        coords, n = getDens('n_' + spec)
        coords, ux = getU('ux' + '_' + spec)
        coords, uy = getU('uy' + '_' + spec)
        coords, uz = getU('uz' + '_' + spec)

        bx = bx / np.sqrt(n*mu0*mu)
        by = by / np.sqrt(n*mu0*mu)
        bz = bz / np.sqrt(n*mu0*mu)

        E = 0.5*(ux**2 + uy**2 + uz**2 + bx**2 + by**2 + bz**2)

        helicity = (ux*bx + uy*by + uz*bz) / E
            
        return coords, helicity

    def getFirehoseThreshold(varid):
        spec = varid[varid.find('_')+1:]
        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame

        coords, pxx = getPress('pxx_' + spec)
        coords, pyy = getPress('pyy_' + spec)
        coords, pzz = getPress('pzz_' + spec)
        coords, pxy = getPress('pxy_' + spec)
        coords, pxz = getPress('pxz_' + spec)
        coords, pyz = getPress('pyz_' + spec)
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        B = np.sqrt(bx**2 + by**2 + bz**2)
        bx = bx / B
        by = by / B
        bz = bz / B

        ppar = pxx*bx**2 + pyy*by**2 + pzz*bz**2 + 2.*(pxy*bx*by + pxz*bx*bz + pyz*by*bz)
        pperp = (pxx+pyy+pzz - ppar) / 2.
        betapar= 2*ppar*self.mu0 / B**2

        firehose =  (pperp - ppar)/ppar + 2./(betapar)
        
        self.params["restFrame"] = tmp
        return coords, firehose

    def getMirrorThreshold(varid):
        spec = varid[varid.find('_')+1:]
        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame

        coords, pxx = getPress('pxx_' + spec)
        coords, pyy = getPress('pyy_' + spec)
        coords, pzz = getPress('pzz_' + spec)
        coords, pxy = getPress('pxy_' + spec)
        coords, pxz = getPress('pxz_' + spec)
        coords, pyz = getPress('pyz_' + spec)
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        B = np.sqrt(bx**2 + by**2 + bz**2)
        bx = bx / B
        by = by / B
        bz = bz / B

        ppar = pxx*bx**2 + pyy*by**2 + pzz*bz**2 + 2.*(pxy*bx*by + pxz*bx*bz + pyz*by*bz)
        pperp = (pxx+pyy+pzz - ppar) / 2.
        betaperp= 2*pperp*self.mu0 / B**2

        mirror =  (ppar - pperp)/ppar + 1./(betaperp)
        
        self.params["restFrame"] = tmp
        return coords, mirror

    def getPTheta(varid):
        spec = varid[varid.find('_')+1:]
        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame
        coords, pxx = getPress('pxx_' + spec)
        coords, pyy = getPress('pyy_' + spec)
        coords, pzz = getPress('pzz_' + spec)
        coords, ux = getU('ux_' + spec)
        coords, uy = getU('uy_' + spec)
        coords, uz = getU('uz_' + spec)

        p = (pxx+pyy+pzz)/3. 

        dims = len(np.shape(ux)) - 1

        dx = np.zeros(dims)
        for d in range(dims):
            dx[d] = coords[d][1] - coords[d][0]
        [duxdx,duxdy,duxdz] = auxFuncs.genGradient(ux,dx)
        [duydx,duydy,duydz] = auxFuncs.genGradient(uy,dx)
        [duzdx,duzdy,duzdz] = auxFuncs.genGradient(uz,dx)

        pTheta = -p*(duxdx + duydy + duzdz)


        self.params["restFrame"] = tmp
        return coords, pTheta

    def getPiD(varid):
        spec = varid[varid.find('_')+1:]
        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame
        coords, pxx = getPress('pxx_' + spec)
        coords, pyy = getPress('pyy_' + spec)
        coords, pzz = getPress('pzz_' + spec)
        coords, pixy = getPress('pxy_' + spec)
        coords, pixz = getPress('pxz_' + spec)
        coords, piyz = getPress('pyz_' + spec)
        coords, ux = getU('ux_' + spec)
        coords, uy = getU('uy_' + spec)
        coords, uz = getU('uz_' + spec)

        p = (pxx+pyy+pzz)/3. 
        pixx = pxx - p
        piyy = pyy - p
        pizz = pzz - p
        
        dims = len(np.shape(ux)) - 1

        dx = np.zeros(dims)
        for d in range(dims):
            dx[d] = coords[d][1] - coords[d][0]
        [duxdx,duxdy,duxdz] = auxFuncs.genGradient(ux,dx)
        [duydx,duydy,duydz] = auxFuncs.genGradient(uy,dx)
        [duzdx,duzdy,duzdz] = auxFuncs.genGradient(uz,dx)

        divu = (duxdx + duydy + duzdz)

        Dxx = (duxdx + duxdx)/2. - divu / 3.
        Dyy = (duydy + duydy)/2. - divu / 3.
        Dzz = (duzdz + duzdz)/2. - divu / 3.
        Dxy = (duxdy + duydx)/2.
        Dxz = (duxdz + duzdx)/2.
        Dyz = (duydz + duzdy)/2.

        piD = -(pixx*Dxx + piyy*Dyy + pizz*Dzz + 2.*(pixy*Dxy + pixz*Dxz + piyz*Dyz))

        self.params["restFrame"] = tmp
        return coords, piD

    def getPiDNormal(varid):
        spec = varid[varid.find('_')+1:]
        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame
        coords, pxx = getPress('pxx_' + spec)
        coords, pyy = getPress('pyy_' + spec)
        coords, pzz = getPress('pzz_' + spec)
        coords, ux = getU('ux_' + spec)
        coords, uy = getU('uy_' + spec)
        coords, uz = getU('uz_' + spec)

        p = (pxx+pyy+pzz)/3. 
        pixx = pxx - p
        piyy = pyy - p
        pizz = pzz - p
        
        dims = len(np.shape(ux)) - 1

        dx = np.zeros(dims)
        for d in range(dims):
            dx[d] = coords[d][1] - coords[d][0]
        [duxdx,duxdy,duxdz] = auxFuncs.genGradient(ux,dx)
        [duydx,duydy,duydz] = auxFuncs.genGradient(uy,dx)
        [duzdx,duzdy,duzdz] = auxFuncs.genGradient(uz,dx)

        divu = (duxdx + duydy + duzdz)

        Dxx = (duxdx + duxdx)/2. - divu / 3.
        Dyy = (duydy + duydy)/2. - divu / 3.
        Dzz = (duzdz + duzdz)/2. - divu / 3.

        piDNormal = -(pixx*Dxx + piyy*Dyy + pizz*Dzz)

        self.params["restFrame"] = tmp
        return coords, piDNormal

    def getPiDShear(varid):
        spec = varid[varid.find('_')+1:]
        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame
        coords, pxx = getPress('pxx_' + spec)
        coords, pyy = getPress('pyy_' + spec)
        coords, pzz = getPress('pzz_' + spec)
        coords, pixy = getPress('pxy_' + spec)
        coords, pixz = getPress('pxz_' + spec)
        coords, piyz = getPress('pyz_' + spec)
        coords, ux = getU('ux_' + spec)
        coords, uy = getU('uy_' + spec)
        coords, uz = getU('uz_' + spec)

        p = (pxx+pyy+pzz)/3. 
        pixx = pxx - p
        piyy = pyy - p
        pizz = pzz - p
        
        dims = len(np.shape(ux)) - 1

        dx = np.zeros(dims)
        for d in range(dims):
            dx[d] = coords[d][1] - coords[d][0]
        [duxdx,duxdy,duxdz] = auxFuncs.genGradient(ux,dx)
        [duydx,duydy,duydz] = auxFuncs.genGradient(uy,dx)
        [duzdx,duzdy,duzdz] = auxFuncs.genGradient(uz,dx)

        divu = (duxdx + duydy + duzdz)

        Dxy = (duxdy + duydx)/2.
        Dxz = (duxdz + duzdx)/2.
        Dyz = (duydz + duzdy)/2.

        piDShear = -2.*(pixy*Dxy + pixz*Dxz + piyz*Dyz)

        self.params["restFrame"] = tmp
        return coords, piDShear


    def sortDrifts(varid, drift, qnE):
        suf = ['x', 'y', 'z']
        id = varid[varid.find('_')-1]
        if id in suf: #Component drift or work
            comp = suf.index(id)
            nonComp = np.squeeze(np.where(np.arange(3) != comp))
            for d in nonComp:
                drift[d,...] = 0.

        if varid.find('work') >= 0:
            data =  np.sum(qnE*drift,axis=0)
        else:
            if not (id in suf):
                data = np.zeros(np.shape(qnE[0]))
                print('Warning, no component index specified for drift. Returning zeros.')
            else:
                data = np.sum(drift,axis=0)

        return data
    
    def getCurvatureDrift(varid): #Curvature drift energization. Based on Appendix F of Juno et al 2021
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        coords, ex = getGenField('ex')
        coords, ey = getGenField('ey')
        coords, ez = getGenField('ez')
        B = np.sqrt(bx**2 + by**2 + bz**2)
        bx = bx/B; by = by/B; bz = bz/B

        dims = len(np.shape(bx)) - 1

        dx = np.zeros(dims)
        for d in range(dims):
            dx[d] = coords[d][1] - coords[d][0]

        [dbxdx,dbxdy,dbxdz] = auxFuncs.eigthOrderGrad2D(bx,dx)
        [dbydx,dbydy,dbydz] = auxFuncs.eigthOrderGrad2D(by,dx)
        [dbzdx,dbzdy,dbzdz] = auxFuncs.eigthOrderGrad2D(bz,dx)

        #kappa = b . (grad b) 
        kappax = bx*dbxdx + by*dbxdy + bz*dbxdz 
        kappay = bx*dbydx + by*dbydy + bz*dbydz
        kappaz = bx*dbzdx + by*dbzdy + bz*dbzdz

        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame
        spec = varid[varid.find('_')+1:]
        coords, Pperp = getPressPerp(varid)
        coords, Ppar = getPressPar(varid)
        self.params["restFrame"] = tmp
        spec = varid[varid.find('_')+1:]
        specIndex = self.speciesFileIndex.index(spec)
        q = self.q[specIndex]
        coords, n = getDens('n_' + spec)
        drift = np.array([(Ppar - Pperp)*(by*kappaz - bz*kappay) / (q*n*B),\
                     (Ppar - Pperp)*(bz*kappax - bx*kappaz) / (q*n*B),\
                     (Ppar - Pperp)*(bx*kappay - by*kappax) / (q*n*B)])
        qnE = np.array([q*n*ex, q*n*ey, q*n*ez])

        data = sortDrifts(varid, drift, qnE)
        return coords, data

    def getCurvatureDriftv2(varid): #Curvature drift energization. Based on Appendix F of Juno et al 2021
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        coords, ex = getGenField('ex')
        coords, ey = getGenField('ey')
        coords, ez = getGenField('ez')
        B = np.sqrt(bx**2 + by**2 + bz**2)
        bx = bx/B; by = by/B; bz = bz/B

        dims = len(np.shape(bx)) - 1

        dx = np.zeros(dims)
        for d in range(dims):
            dx[d] = coords[d][1] - coords[d][0]

        [dbxdx,dbxdy,dbxdz] = auxFuncs.eigthOrderGrad2D(bx,dx)
        [dbydx,dbydy,dbydz] = auxFuncs.eigthOrderGrad2D(by,dx)
        [dbzdx,dbzdy,dbzdz] = auxFuncs.eigthOrderGrad2D(bz,dx)

        #kappa = b . (grad b) 
        kappax = bx*dbxdx + by*dbxdy + bz*dbxdz 
        kappay = bx*dbydx + by*dbydy + bz*dbydz
        kappaz = bx*dbzdx + by*dbzdy + bz*dbzdz

        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame
        spec = varid[varid.find('_')+1:]
        coords, Ppar = getPressPar(varid)
        self.params["restFrame"] = tmp
        spec = varid[varid.find('_')+1:]
        specIndex = self.speciesFileIndex.index(spec)
        q = self.q[specIndex]
        coords, n = getDens('n_' + spec)
        drift = np.array([Ppar*(by*kappaz - bz*kappay) / (q*n*B),\
                     Ppar*(bz*kappax - bx*kappaz) / (q*n*B),\
                     Ppar*(bx*kappay - by*kappax) / (q*n*B)])
        qnE = np.array([q*n*ex, q*n*ey, q*n*ez])

        data = sortDrifts(varid, drift, qnE)
        return coords, data

    def getCurvatureDriftv0(varid): #Guiding center curvature drift energization. 
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        coords, ex = getGenField('ex')
        coords, ey = getGenField('ey')
        coords, ez = getGenField('ez')
        B = np.sqrt(bx**2 + by**2 + bz**2)
        bx = bx/B; by = by/B; bz = bz/B

        dims = len(np.shape(bx)) - 1

        dx = np.zeros(dims)
        for d in range(dims):
            dx[d] = coords[d][1] - coords[d][0]

        [dbxdx,dbxdy,dbxdz] = auxFuncs.eigthOrderGrad2D(bx,dx)
        [dbydx,dbydy,dbydz] = auxFuncs.eigthOrderGrad2D(by,dx)
        [dbzdx,dbzdy,dbzdz] = auxFuncs.eigthOrderGrad2D(bz,dx)

        #kappa = b . (grad b) 
        kappax = bx*dbxdx + by*dbxdy + bz*dbxdz 
        kappay = bx*dbydx + by*dbydy + bz*dbydz
        kappaz = bx*dbzdx + by*dbzdy + bz*dbzdz

        tmp = self.params["restFrame"]
        self.params["restFrame"] = 0 #Not computed in the rest frame
        spec = varid[varid.find('_')+1:]
        coords, Ppar = getPressPar(varid)
        self.params["restFrame"] = tmp
        spec = varid[varid.find('_')+1:]
        specIndex = self.speciesFileIndex.index(spec)
        q = self.q[specIndex]
        coords, n = getDens('n_' + spec)
        drift = np.array([Ppar*(by*kappaz - bz*kappay) / (q*n*B),\
                     Ppar*(bz*kappax - bx*kappaz) / (q*n*B),\
                     Ppar*(bx*kappay - by*kappax) / (q*n*B)])
        qnE = np.array([q*n*ex, q*n*ey, q*n*ez])

        data = sortDrifts(varid, drift, qnE)
        return coords, data

    def getBetatronDrift(varid): #Betatron drift energization. Based on Appendix F of Juno et al 2021
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        coords, ex = getGenField('ex')
        coords, ey = getGenField('ey')
        coords, ez = getGenField('ez')
        B = np.sqrt(bx**2 + by**2 + bz**2)
        bx = bx/B; by = by/B; bz = bz/B

        dims = len(np.shape(bx)) - 1

        dx = np.zeros(dims)
        for d in range(dims):
            dx[d] = coords[d][1] - coords[d][0]

        [dbxdx,dbxdy,dbxdz] = auxFuncs.eigthOrderGrad2D(bx,dx)
        [dbydx,dbydy,dbydz] = auxFuncs.eigthOrderGrad2D(by,dx)
        [dbzdx,dbzdy,dbzdz] = auxFuncs.eigthOrderGrad2D(bz,dx)

        #kappa = b . (grad b) 
        kappax = bx*dbxdx + by*dbxdy + bz*dbxdz 
        kappay = bx*dbydx + by*dbydy + bz*dbydz
        kappaz = bx*dbzdx + by*dbzdy + bz*dbzdz

        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame
        spec = varid[varid.find('_')+1:]
        coords, Pperp = getPressPerp(varid)
        self.params["restFrame"] = tmp
        spec = varid[varid.find('_')+1:]
        specIndex = self.speciesFileIndex.index(spec)
        q = self.q[specIndex]
        coords, n = getDens('n_' + spec)
        drift = np.array([-Pperp*(by*kappaz - bz*kappay) / (q*n*B),\
                     -Pperp*(bz*kappax - bx*kappaz) / (q*n*B),\
                     -Pperp*(bx*kappay - by*kappax) / (q*n*B)])
        qnE = np.array([q*n*ex, q*n*ey, q*n*ez])

        data = sortDrifts(varid, drift, qnE)
        return coords, data

    def getGradBDrift(varid):  #GradB drift energization. Based on Appendix F of Juno et al 2021
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        coords, ex = getGenField('ex')
        coords, ey = getGenField('ey')
        coords, ez = getGenField('ez')
        B = np.sqrt(bx**2 + by**2 + bz**2)
        bx = bx/B; by = by/B; bz = bz/B
        dims = len(np.shape(bx)) - 1

        dx = np.zeros(dims)
        for d in range(dims):
            dx[d] = coords[d][1] - coords[d][0]
        [dbxdx,dbxdy,dbxdz] = auxFuncs.eigthOrderGrad2D(bx/B,dx)
        [dbydx,dbydy,dbydz] = auxFuncs.eigthOrderGrad2D(by/B,dx)
        [dbzdx,dbzdy,dbzdz] = auxFuncs.eigthOrderGrad2D(bz/B,dx)

        cBx = dbzdy - dbydz; cBy = dbxdz - dbzdx; cBz = dbydx - dbxdy; #curl (B / B^2)
        bdotcB = bx*cBx + by*cBy + bz*cBz #b . curl (B / B^2), parallel component
        cBx = cBx - bx*bdotcB; cBy = cBy - by*bdotcB; cBz = cBz - bz*bdotcB #Perp component only

        spec = varid[varid.find('_')+1:]
        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame    
        coords, Pperp = getPressPerp(varid)
        self.params["restFrame"] = tmp
        specIndex = self.speciesFileIndex.index(spec)
        q = self.q[specIndex]
        coords, n = getDens('n_' + spec)

        drift = np.array([Pperp*cBx / (q*n),Pperp*cBy / (q*n),Pperp*cBz / (q*n)])
        qnE = np.array([q*n*ex, q*n*ey, q*n*ez])

        data = sortDrifts(varid, drift, qnE)

        return coords, data

    def getMagnetizationDrift(varid):  #Magnetization drift energization. Based on Appendix F of Juno et al 2021
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        coords, ex = getGenField('ex')
        coords, ey = getGenField('ey')
        coords, ez = getGenField('ez')
        B = np.sqrt(bx**2 + by**2 + bz**2)
        bx = bx/B; by = by/B; bz = bz/B

        dims = len(np.shape(bx)) - 1
    
        dx = np.zeros(dims)
        for d in range(dims):
            dx[d] = coords[d][1] - coords[d][0]
            
        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame
        spec = varid[varid.find('_')+1:]
        coords, Pperp = getPressPerp(varid)
        self.params["restFrame"] = tmp
        [dMxdx,dMxdy,dMxdz] = auxFuncs.eigthOrderGrad2D(-Pperp*bx/B,dx)
        [dMydx,dMydy,dMydz] = auxFuncs.eigthOrderGrad2D(-Pperp*by/B,dx)
        [dMzdx,dMzdy,dMzdz] = auxFuncs.eigthOrderGrad2D(-Pperp*bz/B,dx)
    
        cMx = dMzdy - dMydz; cMy = dMxdz - dMzdx; cMz = dMydx - dMxdy; #curl (M)
        bdotcM = bx*cMx + by*cMy + bz*cMz #b . curl (M), parallel component
        cMx = cMx - bx*bdotcM; cMy = cMy - by*bdotcM; cMz = cMz - bz*bdotcM #Perp component only

        specIndex = self.speciesFileIndex.index(spec)
        q = self.q[specIndex]
        coords, n = getDens('n_' + spec)
        drift = np.array([cMx / (q*n),  cMy / (q*n),  cMz / (q*n)])
        qnE = np.array([q*n*ex, q*n*ey, q*n*ez])

        data = sortDrifts(varid, drift, qnE)

        return coords, data
    
    def getDiamagneticDrift(varid):  #Diamagnetic drift energization. Based on Appendix F of Juno et al 2021
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        coords, ex = getGenField('ex')
        coords, ey = getGenField('ey')
        coords, ez = getGenField('ez')
        B = np.sqrt(bx**2 + by**2 + bz**2)
        bx = bx/B; by = by/B; bz = bz/B

        dims = len(np.shape(bx)) - 1
         
        dx = np.zeros(dims)
        for d in range(dims):
            dx[d] = coords[d][1] - coords[d][0]
            
        tmp = self.params["restFrame"]
        self.params["restFrame"] = 1 #Must be computed in the rest frame
        spec = varid[varid.find('_')+1:]
        coords, Pperp = getPressPerp(varid)
        '''
        coords, pxx = getPress('pxx_' + spec)
        coords, pyy = getPress('pyy_' + spec)
        coords, pzz = getPress('pzz_' + spec)

        p = (pxx+pyy+pzz)/3. 
        Pperp = p
        '''
        self.params["restFrame"] = tmp
        [dpdx,dpdy,dpdz] = auxFuncs.eigthOrderGrad2D(Pperp,dx)
        
        specIndex = self.speciesFileIndex.index(spec)
        q = self.q[specIndex]
        coords, n = getDens('n_' + spec)
        drift = np.array([ (by*dpdz - bz*dpdy) / (q*n*B),\
                               (bz*dpdx - bx*dpdz) / (q*n*B),\
                               (bx*dpdy - by*dpdx) / (q*n*B)])
        qnE = np.array([q*n*ex, q*n*ey, q*n*ez])

        data = sortDrifts(varid, drift, qnE)

        return coords, data

    def getAgyrotropicDrift(varid):  #Agyrotropic drift energization. Based on Appendix F of Juno et al 2021
        coords, bx = getGenField('bx')
        coords, by = getGenField('by')
        coords, bz = getGenField('bz')
        coords, ex = getGenField('ex')
        coords, ey = getGenField('ey')
        coords, ez = getGenField('ez')
        B = np.sqrt(bx**2 + by**2 + bz**2)
        bx = bx/B; by = by/B; bz = bz/B
       
        if self.model == 'pkpm':
            return coords, np.zeros_like(bx)
        else:
            spec = varid[varid.find('_')+1:]
            tmp = self.params["restFrame"]
            self.params["restFrame"] = 1 #Must be computed in the rest frame
            coords, Pperp = getPressPerp(varid)
            coords, Ppar = getPressPar(varid)
            coords, pxx = getPress('pxx_' + spec)
            coords, pyy = getPress('pyy_' + spec)
            coords, pzz = getPress('pzz_' + spec)
            coords, pxy = getPress('pxy_' + spec)
            coords, pxz = getPress('pxz_' + spec)
            coords, pyz = getPress('pyz_' + spec)     
            self.params["restFrame"] = tmp
            
            p = (pxx+pyy+pzz)/3. 
            pixx = pxx - Pperp - (Ppar - Pperp)*bx*bx;
            piyy = pyy - Pperp - (Ppar - Pperp)*by*by;
            pizz = pzz - Pperp - (Ppar - Pperp)*bz*bz;
            pixy = pxy - (Ppar - Pperp)*bx*by;
            pixz = pxz - (Ppar - Pperp)*bx*bz;
            piyz = pyz - (Ppar - Pperp)*by*bz;
                
            dims = len(np.shape(bx)) - 1
          
            dx = np.zeros(dims)
            for d in range(dims):
                dx[d] = coords[d][1] - coords[d][0]
            [dpixxdx,dpixxdy,dpixxdz] = auxFuncs.eigthOrderGrad2D(pixx,dx)
            [dpiyydx,dpiyydy,dpiyydz] = auxFuncs.eigthOrderGrad2D(piyy,dx)
            [dpizzdx,dpizzdy,dpizzdz] = auxFuncs.eigthOrderGrad2D(pizz,dx)
            [dpixydx,dpixydy,dpixydz] = auxFuncs.eigthOrderGrad2D(pixy,dx)
            [dpixzdx,dpixzdy,dpixzdz] = auxFuncs.eigthOrderGrad2D(pixz,dx)
            [dpiyzdx,dpiyzdy,dpiyzdz] = auxFuncs.eigthOrderGrad2D(piyz,dx)


            specIndex = self.speciesFileIndex.index(spec)
            q = self.q[specIndex]
            coords, n = getDens('n_' + spec)
            drift = np.array([ (by*(dpixzdx + dpiyzdy + dpizzdz) - bz*(dpixydx + dpiyydy + dpiyzdz)) / (q*n*B),\
                                   (bz*(dpixxdx + dpixydy + dpixzdz) - bx*(dpixzdx + dpiyzdy + dpizzdz)) / (q*n*B),\
                                   (bx*(dpixydx + dpiyydy + dpiyzdz) - by*(dpixxdx + dpixydy + dpixzdz)) / (q*n*B)])
            qnE = np.array([q*n*ex, q*n*ey, q*n*ez])

            data = sortDrifts(varid, drift, qnE)
            return coords, data




    ###################################################################
    #End read functions
    #################################################################### 
    
    if varidGlobal[0:4] == 'dist':
        coords, data = getDist(varidGlobal)
    elif  varidGlobal.find('drift') >= 0:
        if varidGlobal[0:3] == 'exb':
            coords, data = getExB(varidGlobal)
        elif varidGlobal[0:9] == 'curvature':
            if varidGlobal.find('v2') >= 0:

                coords, data = getCurvatureDriftv2(varidGlobal)
            elif varidGlobal.find('v0') >= 0:
                coords, data = getCurvatureDriftv0(varidGlobal)
            else:
                coords, data = getCurvatureDrift(varidGlobal)
        elif varidGlobal[0:5] == 'gradb':
            coords, data = getGradBDrift(varidGlobal)
        elif varidGlobal[0:6] == 'diamag':
            coords, data = getDiamagneticDrift(varidGlobal)
        elif varidGlobal[0:3] == 'mag':
            coords, data = getMagnetizationDrift(varidGlobal)
        elif varidGlobal[0:5] == 'agyro':
            coords, data = getAgyrotropicDrift(varidGlobal)
        elif varidGlobal[0:4] == 'beta':
            coords, data = getBetatronDrift(varidGlobal)
    elif varidGlobal[0:4] == 'beta':
        coords, data = getBeta(varidGlobal)
    elif varidGlobal[0] == 'b': #magnetic fields
        if self.model == 'vp':
            coords, data = getBVP(varidGlobal)
        else:
            coords, data = getGenField(varidGlobal)
    elif varidGlobal[0] == 'e':
        if varidGlobal[1:4] == 'par':
            coords, data = getEpar(varidGlobal)
        elif self.model == 'vp':
            coords, data = getEVP(varidGlobal)
        else:
            coords, data = getGenField(varidGlobal)
    elif varidGlobal[0:3] == 'div':
        if varidGlobal[3] == 'e':
            coords, data = getDivE(varidGlobal)
        else:
            coords, data = getDivB(varidGlobal)
    elif varidGlobal[0] == 'n': #density
        coords, data = getDens(varidGlobal)
    elif varidGlobal[0] == 'u': #flow velocity
        coords, data = getU(varidGlobal)
    elif varidGlobal[0:3] == 'psi': #2D flux function
        coords, data = getPsi(varidGlobal)
    elif varidGlobal[0:3] == 'phi': #2D potential function
        if self.model == 'vp':
            coords, data = getGenField(varidGlobal)
        else:
            coords, data = getPhi(varidGlobal)
    elif varidGlobal[0:6] == 'stream': #2D stream function
        coords, data = getStream(varidGlobal)
    elif varidGlobal[0] == 'p': #pressure
        if varidGlobal[1:4] == 'par': #Parallel pressure
            coords, data = getPressPar(varidGlobal)
        elif varidGlobal[1:5] == 'perp': #Perp Pressure
            coords, data = getPressPerp(varidGlobal)
        elif varidGlobal[0:8] == 'poynting':#Poynting vector
            coords, data = getPoynting(varidGlobal)
        elif varidGlobal[0:6] == 'ptheta':#p div u
            coords, data = getPTheta(varidGlobal)
        elif varidGlobal[0:3] == 'pid':#piD
            if varidGlobal[0:9] == 'pidnormal':#piDNormal, Cassak PoP 29, 122306 (2022)
                coords, data = getPiDNormal(varidGlobal)
            elif varidGlobal[0:8] == 'pidshear':#piDShear, Cassak PoP 29, 122306 (2022)
                coords, data = getPiDShear(varidGlobal)
            else:
                coords, data = getPiD(varidGlobal)
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
    elif varidGlobal[0] == 'q':
        coords, data = getHeatFlux(varidGlobal)
    elif varidGlobal[0:3] == 'mag':
        if varidGlobal[3] == 'e': #|E|
            coords, data = getMagE(varidGlobal)
        else:# varidGlobal[3] == 'b': #|B|
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
    elif varidGlobal[0:13] == 'crosshelicity':
        coords, data = getCrossHelicity(varidGlobal)
    elif varidGlobal[0:8] == 'firehose':
        coords, data = getFirehoseThreshold(varidGlobal)
    elif varidGlobal[0:6] == 'mirror':
        coords, data = getMirrorThreshold(varidGlobal)
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
    self.data = data
    
    getSlice.postSlice(self)
   
   
