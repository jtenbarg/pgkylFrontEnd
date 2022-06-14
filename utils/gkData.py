import numpy as np
import postgkyl as pg
from utils import getConst
from utils import getData
from utils import getDispersionRelation
from copy import copy, deepcopy

class gkData:
    def __init__(self,filenameBase,fileNum,suffix,varid,params):
        self.filenameBase = filenameBase
        self.fileNum = fileNum
        self.suffix = suffix
        self.varid = varid
        self.params = params

        self.basis = ''
        self.model = ''
        self.dimsX = 1
        self.dimsV = 0
        self.dx = []
        self.po = 0
        self.data = []
        self.coords = []
        self.const = []
        self.max = 0.
        self.min = 0.
        self.time = 0.

        
        self.speciesFileIndex = []
        self.mu0 = 1.
        self.eps0 = 1.
        self.mu = []
        self.q = []
        self.temp = []
        self.beta = []
        self.vA = []
        self.vt = []
        self.c = []
        self.omegaC = []
        self.omegaP = []
        self.d = []
        self.rho = []
        self.debye = []

        if (isinstance(self.params.get('lowerLimits'), type(None))): #Default limits
            self.params["lowerLimits"] = [-1e10, -1e10, -1e10, -1e10, -1e10, -1e10]
        if (isinstance(self.params.get('upperLimits'), type(None))): #Default limits
            self.params["upperLimits"] = [1e10, 1e10, 1e10, 1e10, 1e10, 1e10]
        if (isinstance(self.params.get('axesNorm'), type(None))): #Default axes normalization
            self.params["axesNorm"] = [1., 1., 1., 1., 1., 1.]
        if (isinstance(self.params.get('axesLabels'), type(None))): #Default axes lables
            self.params["axesLabels"] = ['z0', 'z1', 'z2']
        if (isinstance(self.params.get('restFrame'), type(None))): #Default restFrame = false if not specified
            self.params["restFrame"] = 0
        if (isinstance(self.params.get('absVal'), type(None))): #Default absVal = false if not specified
            self.params["absVal"] = 0
        if (isinstance(self.params.get('log'), type(None))): #Default log = false if not specified
            self.params["log"] = 0
        if (isinstance(self.params.get('logThresh'), type(None))): #Default logThresh = 0 if not specified
            self.params["logThresh"] = 0
        if (isinstance(self.params.get('sub0'), type(None))): #Default sub0 = false if not specified
            self.params["sub0"] = 0
        if (isinstance(self.params.get('div0'), type(None))): #Default log0 = false if not specified
            self.params["div0"] = 0
        if (isinstance(self.params.get('timeNorm'), type(None))): #Default time normalization if not specified
            self.params["timeNorm"] = 1
        if (isinstance(self.params.get('symBar'), type(None))): #Default symmetric bar plot
            self.params["symBar"] = 0
        if (isinstance(self.params.get('axisEqual'), type(None))): #Default plot axes
            self.params["axisEqual"] = 0
        if (isinstance(self.params.get('plotContours'), type(None))): #Default cont plot
            self.params["plotContours"] = 0
        if (isinstance(self.params.get('displayTime'), type(None))): #Default display time
            self.params["displayTime"] = 0
        if (isinstance(self.params.get('colormap'), type(None))): #Default colormap
            self.params["colormap"] = 'inferno'

        self.varid = self.varid.lower()  #Convert varid to all lower case, just in case
            
        #if not (isinstance(self.params.get('varid'), type(None))): #Convert varid to all lower case, just in case
            
        getConst.setConst(self)

    def __copy__(self):
        return type(self)(self.filenameBase,self.fileNum,self.suffix,self.varid,self.params)
    
    def __deepcopy__(self, memo): # memo is a dict of id's to copies
        id_self = id(self)        # memoization avoids unnecesary recursion
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                deepcopy(self.filenameBase, memo), 
                deepcopy(self.fileNum, memo),
                deepcopy(self.suffix, memo),
                deepcopy(self.varid, memo),
                deepcopy(self.params, memo))
            memo[id_self] = _copy 
        return _copy

    def getCopy(self):
        newData = deepcopy(self)
        newData.time = self.time
        newData.data = self.data.copy()
        newData.coords = self.coords.copy()
        newData.dx = self.dx.copy()
        return newData
        
    def setMaxMin(self):
        self.max = np.max(self.data)
        self.min = np.min(self.data)

           
    def readData(self):
        if self.varid == 'dispersion':
            self.coords, self.data = getDispersionRelation.getData(self.filenameBase+'frequencies.bp')
        else:
            getData.getData(self)
            self.data = np.squeeze(self.data)
            
            if self.params["sub0"] or self.params["div0"]:
                tmp = copy(self)
                tmp.fileNum = 0
                getData.getData(tmp)
                if self.params["sub0"]:
                    self.data = self.data - tmp.data
                if self.params["div0"]:
                    self.data = self.data / tmp.data
       
            if self.params["absVal"]:
                self.data = np.absolute(self.data)
            if self.params["log"]:
                self.data = np.log10(self.data)
                if self.params["logThresh"] != 0:
                    ii = np.where(self.data < np.max(self.data) + self.params["logThresh"])
                    self.data[ii] =  self.params["logThresh"]
          
            self.setMaxMin()

    def compactRead(self):
        self.readData()
        return self
        
    def __add__(self, *vals):
        newData = self.getCopy()
        for b in vals:
            if isinstance(b, gkData):
                newData.data = self.data + b.data
            else:
                newData.data = self.data + b
        newData.setMaxMin()
        return newData

    def __sub__(self, *vals):
        newData = self.getCopy()

        for b in vals:
            if isinstance(b, gkData):
                newData.data = self.data - b.data
            else:
                newData.data = self.data - b
        newData.setMaxMin()
        return newData

    def __mul__(self, *vals):
        newData = self.getCopy()
        for b in vals:
            if isinstance(b, gkData):
                newData.data = np.multiply(self.data, b.data)
            else:
                newData.data = b*self.data
        newData.setMaxMin()
        return newData

    def __truediv__(self, *vals):
        newData = self.getCopy()
        for b in vals:
            if isinstance(b, gkData):
                newData.data = np.divide(self.data, b.data)
            else:
                newData.data = self.data / b
        newData.setMaxMin()
        return newData

    def __pow__(self, *vals):
        newData = self.getCopy()
        for b in vals:
            if isinstance(b, gkData):
                newData.data = np.power(self.data, b.data)
            else:
                newData.data = np.power(self.data, b)
        newData.setMaxMin()
        return newData

    def integrate(self, axis=None):
        newData = self.getCopy()
        
        if axis is None:
            dims = len(np.shape(np.squeeze(newData.data)))
            newData.data = np.squeeze(np.sum(newData.data))
            dx = 1.
            for d in reversed(range(dims)):
                dx = dx*newData.dx[d]
                del newData.coords[d]
                del newData.params["axesNorm"][d]
        else:
            newData.data = np.squeeze(np.sum(newData.data, axis=axis))
            if isinstance(axis, int):
                dx = newData.dx[axis]
                del newData.coords[axis]
                del newData.params["axesNorm"][axis]
            else:
                axisSort = sorted(axis, reverse=True)
                dx = 1.
                for ax in axisSort:
                    dx *= newData.dx[ax]
                    del newData.coords[ax]
                    del newData.params["axesNorm"][ax]
                    
       
        newData.data = newData.data*dx
        newData.setMaxMin()
        return newData
       
         
  
