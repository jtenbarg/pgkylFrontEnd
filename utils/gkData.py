import numpy as np
import postgkyl as pg
from utils import getConst
from utils import getData
from copy import copy, deepcopy

class gkData:
    def __init__(self,filenameBase,fileNum,suffix,params):
        self.filenameBase = filenameBase
        self.fileNum = fileNum
        self.suffix = suffix
        self.params = params

        self.basis = ''
        self.model = ''
        self.dimsX = 1
        self.dimsV = 0
        self.dx = []
        self.po = 0
        self.varid = []
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
        if (isinstance(self.params.get('axesNorm'), type(None))): #Default limits
            self.params["axesNorm"] = [1., 1., 1., 1., 1., 1.]
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
        if (isinstance(self.params.get('colormap'), type(None))): #Default colormap
            self.params["colormap"] = 'inferno'

            
        if not (isinstance(self.params.get('varid'), type(None))): #Convert varid to all lower case, just in case
            self.params["varid"] = self.params["varid"].lower()
        
        getConst.setConst(self)

    def __copy__(self):
        return type(self)(self.filenameBase,self.fileNum,self.suffix,self.params)
    
    def __deepcopy__(self, memo): # memo is a dict of id's to copies
        id_self = id(self)        # memoization avoids unnecesary recursion
        _copy = memo.get(id_self)
        if _copy is None:
            _copy = type(self)(
                deepcopy(self.filenameBase, memo), 
                deepcopy(self.fileNum, memo),
                deepcopy(self.suffix, memo),
                deepcopy(self.params, memo))
            memo[id_self] = _copy 
        return _copy

    def getCopy(self):
        newData = deepcopy(self)
        newData.time = self.time
        newData.coords = self.coords
        newData.dx = self.dx
        return newData
        
    def setMaxMin(self):
        self.max = np.max(self.data)
        self.min = np.min(self.data)

           
    def readData(self):
        getData.getData(self)
       
        if self.params["sub0"] or self.params["div0"]:
            tmp = copy(self)
            tmp.fileNum = '0'
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

    def integrate(self, axis=0):
        self.data = np.squeeze(np.sum(self.data, axis=axis))
        if isinstance(axis, int):
            dx = self.dx[axis]
            del self.coords[axis]
        else:
            axisSort = sorted(axis, reverse=True)
            dx = 1.
            for ax in axisSort:
                dx *= self.dx[ax]
                del self.coords[ax]
        
        self.data = self.data*dx
        self.coords = np.squeeze(self.coords)
        self.setMaxMin()
       
         
  
