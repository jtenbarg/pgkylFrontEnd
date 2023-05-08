import numpy as np
import matplotlib.pyplot as plt
from utils import gkData
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["text.usetex"] = True

def gkPlot(self,show=1,save=0):

    dims = len(np.shape(self.data))
    axNorm = self.params["axesNorm"]

    saveFilename = self.filenameBase + self.varid + '_'
    plt.figure(figsize=(12,8))
    if dims == 0 or dims > 2:
        raise RuntimeError("You have confused me! {0}D data cannot be plotted.".format(dims))
    if dims == 1:

        plt.plot(self.coords[0]/axNorm[0], self.data, 'k', linewidth=2)
        #plt.plot(self.coords[0]/axNorm[0], np.zeros_like(self.data), 'k', linewidth=1)
        plt.xlabel(self.params["axesLabels"][0])
        #plt.ylabel(self.params["axesLabels"][1])
        plt.autoscale(enable=True, axis='both', tight=True)
        #plt.ylim(-0.0001,0.0001)
        
               
    elif dims == 2:
        #plt.contourf(self.coords[0]/axNorm[0], self.coords[1]/axNorm[1], np.transpose(self.data),200)
        plt.pcolormesh(self.coords[0]/axNorm[0], self.coords[1]/axNorm[1], np.transpose(self.data), shading="gouraud")
        plt.xlabel(self.params["axesLabels"][0])
        plt.ylabel(self.params["axesLabels"][1])
        plt.colorbar()
        plt.set_cmap(self.params["colormap"])
        #plt.clim(-0.08403, 0.1439)
        #Make colorbar symmetric about 0
        if self.params["symBar"]:
            maxLim = max(abs(self.max), abs(self.min))
            plt.clim(-maxLim, maxLim)
            plt.clim(-1.33, 1.33)
            #plt.gci().colorbar.set_clim(-maxLim, maxLim)
            #plt.gca().colorbar.set_clim(-maxLim, maxLim)
       
        #Add optional contours
        if self.params["plotContours"] & (not self.varid[0:4] == 'dist'):
            saveFilename = saveFilename + self.params["varidContours"] + '_'
            cont = gkData.gkData(self.paramFile,self.fileNum,self.suffix,self.params["varidContours"],self.params).compactRead()
            cont.readData()
            plt.rcParams['contour.negative_linestyle'] = 'solid'
            plt.contour(self.coords[0]/axNorm[0], self.coords[1]/axNorm[1], np.transpose(cont.data),\
                            self.params["numContours"], colors = self.params["colorContours"], linewidths=0.75)
        if self.params["axisEqual"]:
            plt.gca().set_aspect('equal', 'box')

   
        
    if self.params["displayTime"] & (self.suffix != '.gkyl'):
        plt.title(r"$t={:.4f}$".format(self.time * self.params["timeNorm"]) + self.params["timeLabel"], loc='right')
    
    if save:
        saveFilename = saveFilename + format(self.fileNum, '04') + '.png'
        plt.savefig(saveFilename, dpi=300, bbox_inches='tight')
        print('Figure written to ',saveFilename)
    if show:
        plt.show()
    plt.close()
