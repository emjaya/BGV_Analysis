##
# @author Colin Barschel (colin.barschel@cern.ch)
# @file various tools for io and functions
#
import decimal
from array import array
from scipy import arange, exp
import numpy as np
import scipy as sp
#from scipy.interpolate import interp1d
from scipy import arange
import pylab as P
import matplotlib as mpl
mpl.interactive(True)
import matplotlib.pyplot as plt
import bz2,gzip,copy
import os.path,time
import calendar
import types
from time import mktime,gmtime
from datetime import datetime
from scipy.misc import factorial
try:
    import cPickle as pickle
except:
    import pickle
import sys
import ROOT
from root_numpy import tree2array

if len(sys.argv) > 1: fname = sys.argv[1]
# tools for plots
prop = mpl.font_manager.FontProperties(size='small')

errampl = 1.

def read_root2numpy(fname,ignore=[],entries=-1):
    # import ROOT
    # from root_numpy import tree2array

    f = ROOT.TFile.Open(fname)
    l=f.GetListOfKeys()
    t = ROOT.TTree()
    f.GetObject(l.First().GetName()+"/SimpleTuple", t)

    d = {}
    branches = []
    for branch in t.GetListOfBranches():
        brname = branch.GetName()
        branches.append(brname)


    # change key names to match old names

    nkey = {
    'Event'        :'event',
    'Run'          :'run',

    'l0EventID'    :'l0evtID',
    'bunchCounter' :'bcid',
    'Tell1ID'      :'tell1ID',
    'nstrips'      :'nstrips',
    'stripADC'     :'stripADC',
    'channelADC'   :'channelADC',
    'cmsChannelADC':'cmsChannelADC',
    'IsRSensor'    :'isrsensor',
    'IsPhiSensor'  :'isphisensor',
    'SensorID'     :'sensorID'

    }

    ntype = {
    'Run'          :np.uint32,
    'Event'        :np.uint32,
    'l0EventID'    :np.uint32,
    'bunchCounter' :np.int16,
    'Tell1ID'      :np.int16,
    'nstrips'      :np.int16,
    'stripADC'     :np.float32,
    'channelADC'   :np.float32,
    'cmsChannelADC':np.float32,
    'IsRSensor'    :np.int16,
    'IsPhiSensor'  :np.int16,
    'SensorID'     :np.int16

    }


    for branch in t.GetListOfBranches():
        brname = branch.GetName()
        if brname in ignore: continue
        if brname not in nkey: continue

        print '-> keep',brname,'->',nkey[brname]
        if brname in ['channelADC', 'cmsChannelADC', 'stripADC' ]:
            d[nkey[brname]] = tree2array(t, branches=[brname],start=0,stop=entries)
        else: 
            d[nkey[brname]] = np.array(tree2array(t, branches=[brname],start=0,stop=entries)[brname],dtype=ntype[brname])

    return d

ignore = ['isrsensor',
          'isphisensor'
          'sensorID']

tree = read_root2numpy(fname,ignore=ignore,entries=-1)


linksaverage = {}
linkssigma = {}

for i in range(64):
    linksaverage[i] = []#*32 #create key and list with 32 entries per list, equalling to 32 channels per link
    linkssigma[i]   = []#*32

# len(tree['cmsChannelADC'][0][0]) = 2048
# len(tree['cmsChannelADC']) = 25997
# len(tree['cmsChannelADC'][0]) = 1
"""
for evt in range(1000, len(tree['channelADC'])):
    for i in range(64):
        linksaverage[i].append(np.average(tree['channelADC'][evt][0][(32*i):(32+32*i)]))
    linksaverage[i].append()
    linkssigma[i].append(evt)


for i in range(25):
    for j in range(5):
        linksorted[63][i+j*25]=linksaverage[63][(i*5+j)*200:(i*5+j)*200+200]



"""

fig = plt.figure(figsize=(16,7.5), facecolor = 'white')
ax = fig.add_subplot(111)

x = range(len(tree['channelADC'][0][0]))
y = tree['channelADC'][0][0]    
   
    
    
ax.plot(x     = x
        ,y    = y
        ,color='blue'
        ,linestyle='')














