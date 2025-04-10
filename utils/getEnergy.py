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
