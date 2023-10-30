import numpy as np
from utils import gkData
from utils import gkPlot as plt
#End Preamble###############

#Tested to handle g0 and g2: VM, 5M, 10M
#Requires a _params.txt file in your data directory of the form gkeyllOutputBasename_params.txt! See example_params.txt for formatting
paramFile = '/Users/jtenbarg/Desktop/runs/gemEddyv43/Data/gem_params.txt' 

fileNum = 18
varid = 'jz' #See table of choices in README

#End input#########################################

var = gkData.gkData(paramFile,fileNum,varid,{}).compactRead()

plt.gkPlot(var, show=1, save=0) #show and save are optional. Default show=1, save=0. Saves to var.filenameBase directory

