import numpy as np
import adios

def _is_gkyl(file_name : str, offset : int) -> bool:
  magic = np.fromfile(file_name, dtype=np.dtype('b'), count=5, offset=offset)
  if np.array_equal(magic, [103, 107, 121, 108,  48]):
    return True
  else:
    return False

def preSlice(self, filename):
    #Create temporary grid from adios attributes
    if self.suffix == '.bp':
        fh = adios.file(filename)
        lower = np.atleast_1d(adios.attr(fh, 'lowerBounds').value)
        upper = np.atleast_1d(adios.attr(fh, 'upperBounds').value)
        cells = np.atleast_1d(adios.attr(fh, 'numCells').value)
        fh.close()
      
    elif self.suffix == '.gkyl':
        file_type = 1
        version = 0
        dti8 = np.dtype('i8')
        dtf = np.dtype('f8')
        doffset = 8
        offset = 0
        if _is_gkyl(filename, offset):
            offset += 5
            version = np.fromfile(filename, dtype=dti8, count=1, offset=offset)[0]
            offset += 8

            file_type = np.fromfile(filename, dtype=dti8, count=1, offset=offset)[0]
            offset += 8

            meta_size = np.fromfile(filename, dtype=dti8, count=1, offset=offset)[0]
            offset += 8
            # read meta
            offset += meta_size


        # read real-type
        realType = np.fromfile(filename, dtype=dti8, count=1, offset=offset)[0]
        if realType == 1:
            dtf = np.dtype('f4')
            doffset = 4
        #end
        offset += 8
        # read grid dimensions
        num_dims = np.fromfile(filename, dtype=dti8, count=1, offset=offset)[0]
        offset += 8
        # read grid shape
        cells = np.fromfile(filename, dtype=dti8, count=num_dims, offset=offset)
        offset += num_dims*8
        # read lower/upper
        lower = np.fromfile(filename, dtype=dtf, count=num_dims, offset=offset)
        offset += num_dims*doffset
        upper = np.fromfile(filename, dtype=dtf, count=num_dims, offset=offset)
    else:
        #If not g2 or g0 data, read full range
        [zs.append(None) for d in range(6)]
        return zs
    
    dims = len(lower)
    coords = [np.linspace(lower[d], upper[d], cells[d]+1) for d in range(dims)]
    axNorm = self.params["axesNorm"]
    zs = []
    for d in range(6):
        if d >= dims:
            zs.append(None)
        else:
            idxs = np.searchsorted(coords[d], [self.params["lowerLimits"][d]*axNorm[d], \
                                                     self.params["upperLimits"][d]*axNorm[d]])
            if not (self.params["lowerLimits"][d]*axNorm[d] in coords[d]) and idxs[0] != 0:
                idxs[0] -= 1
            if not (self.params["upperLimits"][d]*axNorm[d] in coords[d]) and idxs[0] != 0:
                idxs[1] -= 1  

            #Handle edge cases
            if idxs[0] >= cells[d]: idxs[0] = idxs[0]-2
            if idxs[1] >=  cells[d]: idxs[1] = idxs[1]-1
            if idxs[0] == idxs[1]: idxs[1] += 1
           
            zs.append('{0}:{1}'.format(idxs[0], idxs[1]))

    return zs
    

def postSlice(self):
    dims = len(np.shape(np.squeeze(self.data)))
    coords = self.coords.copy()
    axNorm = self.params["axesNorm"]
    idx = None
    idxValues = [slice(0, self.data.shape[d]) for d in range(dims)]
    #print(self.coords[0]/axNorm[0],self.coords[1]/axNorm[1])
    for d in range(dims):
        
        idxs = np.searchsorted(coords[d], [self.params["lowerLimits"][d]*axNorm[d], \
                                                   self.params["upperLimits"][d]*axNorm[d]])
        

        #Handle edge cases
        if idxs[0] >= len(coords[d]): idxs[0] = idxs[0]-1
        if idxs[0] == idxs[1]: idxs[1] += 1
        idxValues[d] = slice(idxs[0], idxs[1])
        
        self.coords[d] = self.coords[d][idxValues[d]]
    #print(self.coords[0]/axNorm[0],self.coords[1]/axNorm[1])
    self.data = np.squeeze(self.data[tuple(idxValues)])
    axesNorm = self.params["axesNorm"].copy()
    for d in reversed(range(dims)):
        if len(self.coords[d]) == 1:
            del self.coords[d]
            del axesNorm[d]
            self.dx = np.delete(self.dx, d)
    self.params["axesNorm"] = axesNorm
    
    #dims = len(np.shape(np.squeeze(self.data)))
    #if dims == 1:
    #    self.coords = self.coords[0] #Simplifies some later operations to bypass the list format


         
