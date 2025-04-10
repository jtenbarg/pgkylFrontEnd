import numpy as np
import postgkyl as pg

def getData(self):
    fieldEnergyFilename = self.filenameBase + 'field-energy' + self.suffix
    t = np.squeeze(pg.GData(fieldEnergyFilename).get_grid())
    fieldEnergy = pg.GData(fieldEnergyFilename).get_values()

    imoms = []
    for i in range(self.nspec):
        filename = self.filenameBase + self.speciesFileIndex[i] + '-imom' + self.suffix
        imoms.append(pg.GData(filename).get_values())

    return t, fieldEnergy, np.array(imoms)

'''
imoms file contents
PKPM
rho
rhoux
rhouy
rhouz
rhoux^2 = rho * ux^2
rhouy^2
rhouz^2
p_par
p_perp

VM
rho
rhoux
rhouy
rhouz
E = 3*n*T/m + nu^2

DG fluids
rho
rhoux
rhouy
rhouz
rhou^2
p/(gamma-1)
'''