import numpy as np
from utils import gkData
from utils import gkPlot as plt
#End Preamble###############

#Tested to handle g0 and g2: VM, 5M, 10M
#Requires a _params.txt file in your data directory or the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting
paramsFile = '/Users/jtenbarg/Desktop/runs/gemEddyv43/Data/gem_params.txt' 


fileNum = 19
suffix = '.bp'
varid = 'jz' #See table of choices in README

#End input#########################################

var = gkData.gkData(paramsFile,fileNum,suffix,varid,{}).compactRead()

plt.gkPlot(var, show=1, save=0) #show and save are optional. Default show=1, save=0. Saves to var.filenameBase directory

