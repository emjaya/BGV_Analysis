import matplotlib as mpl
#mpl.use('Agg')
mpl.interactive(True)
import numpy as np
import scipy as sp
import pylab as P
import matplotlib.pyplot as plt
import string
import math, os, sys
from time import mktime
from datetime import datetime
from matplotlib.dates import *
import matplotlib.cm as cm
import matplotlib.colors as colors
from mytools import *
import mytools as mt
try:
    import cPickle as pickle
except:
    import pickle
from config import *

beam = 1

if len(sys.argv) > 1:
    fillnr = int(sys.argv[1])
if len(sys.argv) > 2:
    fillnr = int(sys.argv[1])
    beam = int(sys.argv[2])

#### FBCT ####

cvsfile = BCT_DATA_DIR+'/fbct_'+str(fillnr)+'_A'+str(beam)+'.csv.bz2'

print cvsfile

d = readFbctCsv(cvsfile)

for i in range(len(d)):
    if (d[i][76])>1.e+08: # 76 corresponds to an approximately correct timestamp
        print i


#to do: figure out a way to get timestamps from data file and create unix-timerange out of this, in which we're looking for filled bunches.