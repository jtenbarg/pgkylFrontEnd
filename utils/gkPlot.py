import numpy as np
import matplotlib.pyplot as plt
from utils import gkData

def gkPlot(var,show=1,save=0):

    dims = len(np.shape(var.data))
    axNorm = var.params["axesNorm"]

    saveFilename = var.filenameBase + var.params['varid'] + '_'
    plt.figure(figsize=(6,4))
    if dims == 0 or dims > 2:
        raise RuntimeError("You have confused me! {0}D data cannot be plotted.".format(dims))
    if dims == 1:

        plt.plot(var.coords/axNorm[0], var.data, 'k', linewidth=2)
        plt.xlabel(var.params["axesLabels"][0])
        plt.autoscale(enable=True, axis='both', tight=True)
        #plt.rc('axes', labelsize=30)

               
    elif dims == 2:
        plt.contourf(var.coords[0]/axNorm[0], var.coords[1]/axNorm[1], np.transpose(var.data),200)
        #plt.pcolormesh(var.coords[0]/axNorm[0], var.coords[1]/axNorm[1], np.transpose(var.data))
        plt.xlabel(var.params["axesLabels"][0])
        plt.ylabel(var.params["axesLabels"][1])
        plt.colorbar()
        plt.set_cmap(var.params["colormap"])

        #Make colorbar symmetric about 0
        if var.params["symBar"]:
            maxLim = max(abs(var.max), abs(var.min))
            plt.clim(-maxLim, maxLim)
        #Add optional contours
        if var.params["plotContours"] & (not var.params['varid'][0:4] == 'dist'):
            params = var.params
            saveFilename = saveFilename + var.params["varidContours"] + '_'
            params["varid"] = var.params["varidContours"]
            cont = gkData.gkData(var.filenameBase,var.fileNum,var.suffix,params)
            cont.readData()
            plt.rcParams['contour.negative_linestyle'] = 'solid'
            plt.contour(var.coords[0]/axNorm[0], var.coords[1]/axNorm[1], np.transpose(cont.data),\
                            var.params["numContours"], colors = var.params["colorContours"], linewidths=0.5)
        if var.params["axisEqual"]:
            plt.gca().set_aspect('equal', 'box')

   
        
    if var.params["displayTime"] & (var.suffix != '.gkyl'):
        plt.title("$t={:.4f}$".format(var.time * var.params["timeNorm"]) + var.params["timeLabel"], loc='right')
    
    if save:
        saveFilename = saveFilename + format(int(var.fileNum), '04') + '.png'
        plt.savefig(saveFilename, dpi=300)
        print('Figure written to ',saveFilename)
    if show:
        plt.show()
