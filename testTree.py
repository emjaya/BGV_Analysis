import decimal
from array import array
from scipy import arange, exp
import numpy as np
import scipy as sp
#from scipy.interpolate import interp1d
from scipy import arange
import pylab as P
import matplotlib as mpl
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

if len(sys.argv) > 1: fname = sys.argv[1]
# tools for plots
prop = mpl.font_manager.FontProperties(size='small')

errampl = 1.

import ROOT
from root_numpy import tree2array

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
#    'channelADC'   :'channelADC',
#    'cmsChannelADC':'cmsChannelADC',
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
#    'channelADC'   :np.float32,
#    'cmsChannelADC':np.float32,
'IsRSensor'    :np.int16,
'IsPhiSensor'  :np.int16,
'SensorID'     :np.int16

}

ignore = ['isrsensor',
          'isphisensor'
          'sensorID']
entries = -1         


for branch in branches[0:branches.index('stripADC')]:
    brname = branch
    if brname in ignore: continue
    if brname not in nkey: continue

    print '-> keep',brname,'->',nkey[brname]
    d[brname] = np.array(tree2array(t, branches=[brname],start=0,stop=entries)[brname],dtype=ntype[brname])

for branch in branches[branches.index('stripADC'):]:
    brname = branch
    if brname in ignore: continue

    if brname not in nkey: continue

    print '-> keep',brname,'->',nkey[brname]

    for i in range(len((tree2array(t, branches=[brname],start=0,stop=entries)[brname]))):   

    	d[brname]=np.array(tree2array(t, branches=[brname],start=0,stop=entries)[brname][i],dtype=ntype[brname])



