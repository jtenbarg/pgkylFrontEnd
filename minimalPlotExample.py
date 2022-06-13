import numpy as np
from utils import gkData
from utils import gkPlot as plt
params = {} #Initialize dictionary to store plotting and other parameters

#Tested to handle g0 and g2: VM, 5M, 10M
#Include the final underscore or dash in the filename
#Expects a filenameBase + 'params.txt' file to exist! See example_params.txt for the formatting
filenameBase = '/Users/jtenbarg/Desktop/runs/gemEddyv43/Data/gem_' 


fileNum = 19
suffix = '.bp'
varid = 'jz' #See table of choices in README

##########################################

var = gkData.gkData(filenameBase,fileNum,suffix,varid,params).compactRead()

plt.gkPlot(var, show=1, save=0) #show and save are optional. Default show=1, save=0. Saves to filenameBase directory

