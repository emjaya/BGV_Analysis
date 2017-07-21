#!/usr/bin/env python
# inspired from velo2-assembly.py
# Author: Karol Hennessy - karol.hennessy@cern.ch
# Author: Massimiliano Ferro-Luzzi
# changes by Mariana Rihl
# Last edited: 29 November 2016

from Gaudi.Configuration  import *
from GaudiPython.Bindings import AppMgr
from GaudiPython.Bindings import gbl
from Configurables        import LHCbApp
from Configurables       import PrepareVeloFullRawBuffer
from Configurables       import DecodeVeloFullRawBuffer
from Configurables       import createODIN
#from Configurables       import nzsStreamListener
#from Configurables       import LbAppInit
#from Configurables       import Vetra
#from Configurables       import (CondDB, CondDBAccessSvc)
import GaudiPython
from ROOT                 import TH2F, TCanvas, TFile, TH1F
import sys
from math import sqrt
import pickle

##############################
import matplotlib as mpl
mpl.use('Agg')
#mpl.use('Qt4Agg')
mpl.interactive(True)
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from numpy import *
import string
import math, os, sys
#from mytools import *
#     from funcs import *
#from defs import *
#import copy
##############################

#from Configurables       import DataOnDemandSvc
#from DAQSys.DecoderClass import Decoder
#from DAQSys.Decoders     import DecoderDB as ddb

lhcb = LHCbApp()

nevts                  = 1000000000
print 'Starting bgv_raw.py....'
if len(sys.argv)<2 :
   print 'You must provide the run number [and optionally num_events num_TAE]: python BGV_pulseshape.py 1571 [1000 2]' 
   sys.exit(-1)
for i in range(0,len(sys.argv)):
   print 'sys.argv[',i,'] = ',sys.argv[i]
if len(sys.argv)>2 : 
   nevts = int(sys.argv[2])
runnumber              = sys.argv[1]
print 'Will process at most ',nevts,' events from run number ', runnumber

# default: TAE = 0, no next and no prev
firstsampleprev = 3
lastsamplenext = 3
if len(sys.argv)>3:
   TAE = int(sys.argv[3])
   if TAE in [1,2,3]:
      firstsampleprev = 3-TAE
      lastsamplenext  = 3+TAE
   else:
      print 'cannot handle such TAE... I use TAE = 0'

#                0               1               2               3                 4               5               6
fullloc      = [ 'Prev3/'      , 'Prev2/'      , 'Prev1/'      , ''              , 'Next1/'      , 'Next2/'      , 'Next3/'        ]
fullprepList = [ 'preparePrev3', 'preparePrev2', 'preparePrev1', 'prepareCentral', 'prepareNext1', 'prepareNext2', 'prepareNext3'  ]
fulldecoList = [ 'decodePrev3' , 'decodePrev2' , 'decodePrev1' , 'decodeCentral' , 'decodeNext1' , 'decodeNext2' , 'decodeNext3'   ]
loc      = []
prepList = []
decoList = []
countmissing = []
for s in range(firstsampleprev,lastsamplenext+1):  # a subselection 
    loc.append(      fullloc     [s])
    prepList.append( fullprepList[s])
    decoList.append( fulldecoList[s])
    print 'Subselected sample ',s,'  loc: ',fullloc[s],'  prep: ',fullprepList[s],'  deco: ',fulldecoList[s]
    countmissing.append(0)

bcidlist = []
files = []
rootfile = 'none'
if runnumber == '1245':
   rootfile = 'Run_0001245_20151119.root'   
   files.append("DATAFILE='data/Run_0001245_20151119-205814.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
if runnumber == '1378':
   rootfile = 'Run_0001378_20151210.root'
   files.append("DATAFILE='/afs/cern.ch/project/cryoblm/bgv_tmp1/Run_0001378_20151210-171637.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='/afs/cern.ch/project/cryoblm/bgv_tmp1/Run_0001378_20151210-171713.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='/afs/cern.ch/project/cryoblm/bgv_tmp1/Run_0001378_20151210-171748.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='/afs/cern.ch/project/cryoblm/bgv_tmp1/Run_0001378_20151210-171823.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
if runnumber == '1383':
   rootfile = 'Run_0001383_20151211.root'
   files.append("DATAFILE='data/Run_0001383_20151211-164813.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='data/Run_0001383_20151211-164923.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='data/Run_0001383_20151211-165036.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='data/Run_0001383_20151211-165156.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
if runnumber == '1384':
   rootfile = 'Run_0001384_20151214.root'
   files.append("DATAFILE='data/Run_0001384_20151214-142204.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='data/Run_0001384_20151214-142319.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='data/Run_0001384_20151214-142432.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='data/Run_0001384_20151214-142546.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
if runnumber == '1385':
   rootfile = 'Run_0001385_20151214.root'
   files.append("DATAFILE='data/Run_0001385_20151214-144435.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='data/Run_0001385_20151214-144610.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='data/Run_0001385_20151214-144748.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='data/Run_0001385_20151214-144925.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
if runnumber == '1577':
   rootfile  = 'Run_0001577_20160419.root'
   files.append("DATAFILE='/afs/cern.ch/work/m/massi/BGV/analysis/data/Run_0001577_20160419-105252.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='/afs/cern.ch/work/m/massi/BGV/analysis/data/Run_0001577_20160419-111452.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
if runnumber == '1571':
   rootfile = 'Run_0001571_20160417.root'
   files.append("DATAFILE='/afs/cern.ch/work/m/massi/BGV/analysis/data/Run_0001571_20160417-140118.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
if runnumber == '1582':
   rootfile = 'Run_0001582_20160421.root'
   files.append("DATAFILE='/afs/cern.ch/work/m/massi/BGV/analysis/data/Run_0001582_20160421-191640.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
if runnumber == '1583':
   rootfile = 'Run_0001583_20160421.root'
   files.append("DATAFILE='/afs/cern.ch/work/m/massi/BGV/analysis/data/Run_0001583_20160421-191913.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
if runnumber == '1584':
   rootfile = 'Run_0001584_20160425.root'
   files.append("DATAFILE='data/Run_0001584_20160425-183555.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
if runnumber == '1585':
   rootfile = 'Run_0001585_20160425.root'
   files.append("DATAFILE='data/Run_0001585_20160425-184613.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
if runnumber == '1586':
   rootfile = 'Run_0001586_20160425.root'
   files.append("DATAFILE='data/Run_0001586_20160425-190407.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
if runnumber == '1587':
   rootfile = 'Run_0001587_20160425.root'
   files.append("DATAFILE='data/Run_0001587_20160425-191853.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
if runnumber == '1588':
   rootfile = 'Run_0001588_20160427.root'
   files.append("DATAFILE='data/Run_0001588_20160427-000411.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='data/Run_0001588_20160427-000920.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='data/Run_0001588_20160427-001922.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='data/Run_0001588_20160427-001426.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   bcidlist = [2169]
if runnumber == '1589':
   rootfile = 'Run_0001589_20160503.root'
   files.append("DATAFILE='file:///afs/cern.ch/work/m/mrihl/public/Run_0001589_20160427-102746.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='file:///afs/cern.ch/work/m/mrihl/public/Run_0001589_20160427-102853.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='file:///afs/cern.ch/work/m/mrihl/public/Run_0001589_20160427-102957.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   files.append("DATAFILE='file:///afs/cern.ch/work/m/mrihl/public/Run_0001589_20160427-103101.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
   bcidlist = [96,390,590,790,990,2169,2369,2569,2769,3060,3260,3460]

EventSelector().Input  = files
EventSelector().PrintFreq = 500

# init
appMgr      = AppMgr()
appMgr.addAlgorithm('LbAppInit')
for s in range(0,len(prepList)):
    appMgr.addAlgorithm('PrepareVeloFullRawBuffer/'+prepList[s])
    appMgr.addAlgorithm('DecodeVeloFullRawBuffer/'+decoList[s] )

# in TAE events ODIN bank exists only for central event
appMgr.addAlgorithm('createODIN')

prep = []
deco = []
for s in range(0,len(prepList)):
    print 'Added algo prepList[',s,'] to prep'
    prep.append(appMgr.algorithm(prepList[s]))
    deco.append(appMgr.algorithm(decoList[s]))
for s in range(0,len(prepList)):
    print 'Config prep[',s,'] with loc[',s,'] =',loc[s]
    prep[s].RunWithODIN = False
    prep[s].OutputLevel = 3
    prep[s].IgnoreErrorBanks = True
    prep[s].ADCLocation      = loc[s]+'Raw/Velo/ADCBank'
    prep[s].ADCPartialLoc    = loc[s]+'Raw/Velo/PreparedPartialADC'
    prep[s].RawEventLocation = loc[s]+'DAQ/RawEvent'
    print 'Config deco[',s,'] with loc[',s,'] =',loc[s]
    deco[s].CableOrder = [3, 2, 1, 0] 
    deco[s].SectorCorrection = False 
    deco[s].OutputLevel = 3
    deco[s].DecodedPartialADCLocation = loc[s]+'Raw/Velo/DecodedPartialADC'
    deco[s].ADCLocation               = loc[s]+'Raw/Velo/ADCBank'
    deco[s].DecodedADCLocation        = loc[s]+'Raw/Velo/DecodedADC'
    deco[s].DecodedHeaderLocation     = loc[s]+'Raw/Velo/DecodedHeaders'
    deco[s].EventInfoLocation         = loc[s]+'Raw/Velo/EvtInfo' 
    deco[s].DecodedPedestalLocation   = loc[s]+'Raw/Velo/DecodedPed'
for s in range(0,len(prepList)): 
    print 'Prepare algo : loc[',s,'] :   '
    props = prep[s].properties()
    for k in props.keys(): print '      ',k, props[k].value()
    print 'Decoder algo : loc[',s,']'
    props = deco[s].properties()
    for k in props.keys(): print '      ',k, props[k].value()



appMgr.OutputLevel = 3
#prepar.OutputLevel = 3
#decode.OutputLevel = 3

evt         = appMgr.evtsvc()

def get_headerdata_by_key(headers,key):
    for h in headers.containedObjects():
          k = h.key()
          if k == key: 
             hdata = h.decodedData()
             return  hdata 

def get_adcdata_by_key(adcs,key):
    for a in adcs.containedObjects():
          k = a.key()
          if k == key: 
             adcdata = a.decodedData()
             return adcdata 

def get_peds(appMgr,nforpeds,tell1numbers,initpeds,outliercut):
    print '###### Make pedestals '
    peds = {}
    cnts = {}
    for key in tell1numbers: 
        peds[key] = [0.0]*2048
        cnts[key] = [0.0]*2048
    n = 0
    while n <= nforpeds:
          result = appMgr.run(1)
          if  not evt['/Event'].__nonzero__() : 
              print '###### Broken event ???   Event ',n
              break
          for s in range(0,len(loc)):
              adcs    = evt[loc[s]+'Raw/Velo/DecodedADC']
              for sen in adcs.containedObjects():
                  key = sen.key()
                  if key in tell1numbers:
                     sdata = sen.decodedData()
                     for ch in range(2048): 
                         if abs(initpeds[key][ch]-sdata[ch])<outliercut:
                            peds[key][ch] += sdata[ch] 
                            cnts[key][ch] += 1.0
          n += 1  # start counting events at 1
          print 'Ped event ',n

    for key in tell1numbers: 
        for ch in range(2048): 
            peds[key][ch] = peds[key][ch] / cnts[key][ch]

    return peds

def save_obj(obj, name ):
    #with open('obj/'+ name + '.pkl', 'wb') as f:
    with open(pkldir + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    #with open('obj/' + name + '.pkl', 'rb') as f:
    with open(pkldir + name + '.pkl', 'rb') as f:
        return pickle.load(f)

############################ main: ############################################################

pkldir = '/afs/cern.ch/user/m/mrihl/BGVAnalysis/'
skipevts = 0
if skipevts > 0:
    print '###### Skip ', skipevts, ' events'
    appMgr.run(skipevts)

#frawhists = make_raw_histograms(appMgr,nevts,[1])
#odinbcid=justcount_events(appMgr,nevts)

#bgvtell1numbers=[1,2,4,5,6,7,8,9] # 2015: nr 9 is not receiving data from FE
bgvtell1numbers=[1]  
tell1numbers=[1,2,4,5,6,7,8,9]
#frawhists = make_raw_histograms(appMgr,nevts,[1])
#frawhists = make_raw_histograms(appMgr,nevts,[1,2,4,5,6,7,8,9])
#frawhists = make_raw_histograms(appMgr,nevts,bgvtell1numbers,False)

nevtsperstep = 1000
nsteps = 25
do_peds = False
#do_peds = True
if do_peds:
   outliercut = 5.0
   peds = {}
   niter = 5
   nforpeds = nevtsperstep/niter
   for key in bgvtell1numbers: 
       peds[key] = [512.0]*2048
       print 'peds[',key,'][0:5] = ',peds[key][0:5],' ...'
   for i in range(niter):
       peds = get_peds(appMgr,nforpeds,bgvtell1numbers,peds,(niter-i)*outliercut)
       for key in bgvtell1numbers: print 'peds[',key,'][0:5] = ',peds[key][0:5],' ...'
   pedsname = 'peds_'+runnumber+'_v1'
   save_obj(peds,pedsname)
else:
  peds = load_obj('peds_'+runnumber+'_v1')
  appMgr.run(nevtsperstep) # skip first step!
# ---->for key in bgvtell1numbers: 
# ---->    print 'peds[',key,'][0:5] = ',peds[key][0:5],' ...'

channeldatacut = 50

wanted_bcid_list = bcidlist

######################create pulseShapeHistograms##########################

#def pulseShapeHistograms(appMgr,nevtsperstep,nsteps,tell1numbers,wanted_bcid_list,peds,channeldatacut):
print '###### Make pulse shape histograms '
f = TFile('PulseShape_run_'+runnumber+'.root', 'recreate')
channellist = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
#channellist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
linklist = [20,32,40]
#linklist = range(64)
pulshists = {}
completepulsehists = {}
averagepulshists = {}
# stephists = {}
for key in tell1numbers:
    pulshists[key] = []
    completepulsehists[key] = {}
    averagepulshists[key] = {}
#    stephists[key] = []
    print 'Created pulshists[key=',key,'] list'
    for ll in range(0,len(linklist)):
        l = linklist[ll]
        hname = "h_PSscan_source%03d"%(key)+"_link%03d"%(l)
        htitl = "PulseShapeScan source "+str(key)+" link "+str(l)
        nxbi = (nsteps)*len(loc)
        xmax = (nsteps)*(0.5*len(loc))
        xmin = -xmax
        pulshists[key].append( TH2F(hname, htitl, nxbi, xmin,xmax, 150, 372.5,671.5) )
        print 'At index ',len(pulshists[key])-1,'   created histogram ',hname,nxbi,xmin,xmax,150, 372.5,671.5

        completepulsehists[key][l] = {}
        averagepulshists[key][l] = []
        for timeslot in range(nxbi):
          completepulsehists[key][l][xmin+timeslot] = []

    #     for st in range(nsteps):
    #         hname = "h_PSscan_source%03d"%(key)+"_link%03d"%(l)+"_step%02d"%(st)
    #         htitl = "PulseShapeScan source "+str(key)+" link "+str(l)+" step "+str(st)
    #         stephists[key].append( TH2F(hname, htitl, nxbi, xmin,xmax, 150, 372.5,671.5) )
    print 'Stop creating histograms'
ncheck = 0
n = 0
skipped = 0
step = 0
print 'Event ',n,'   step ',step
n_badpcn = 0
not_wanted_bcid = 0
while True:
      n += 1  # start counting events at 1
      if n>1 and n%nevtsperstep == 1: 
         print 'Event ',n,'   got ',nevtsperstep,' events in step ',step,' out of ',nsteps
         step   += 1
         print '      => step changed to ',step

      skipevent = False
      result = appMgr.run(1)
      #print 'Event ',n,' result = ',result
      if  not evt['/Event'].__nonzero__() : 
          print '###### Broken event ???   Event ',n
          break
          #skipevent = True
      if  n > nevtsperstep*nsteps: 
          break
      adcs    = []
      headers = []
      evtinfo = []
      xx = []
      for s in range(0,len(loc)):
          a = evt[loc[s]+'Raw/Velo/DecodedADC']    # e.g. evt['Prev2/Raw/Velo/DecodedADC'][1] is 'previous2' data of bgvtell01 but evt['Prev2/Raw/Velo/DecodedADC'] is defined
          h = evt[loc[s]+'Raw/Velo/DecodedHeaders'] 
          e = evt[loc[s]+'Raw/Velo/EvtInfo']
          xx.append( (step+(s-0.5*len(loc))*nsteps) )
          
          if (not(a) or not(h)):
             print '   MISSING ADCS or HEADERS DATA in loc [',s,'] ',loc[s]+'Raw/Velo/... !!!'
             countmissing[s] += 1
             skipevent = True
          else:
             adcs.append(    a ) 
             headers.append( h )
             evtinfo.append( e )
      odinloc = 'DAQ/ODIN' #odinloc = 'Raw/Velo/EvtInfo'
      odin = evt[odinloc]
      if odin : 
         bcid  =  odin.bunchId()
         bctyp =  odin.bunchCrossingType()
         orbnr =  odin.orbitNumber()
         evtim = (odin.eventTime()).ns()
         evtyp =  odin.eventType()
         if not (bcid in wanted_bcid_list): 
            skipevent = True
            not_wanted_bcid += 1
      if not(skipevent): 
         ncheck += 1
        #TAEfirstPCN = 0
         for key in tell1numbers:
             pedestals = peds[key]
             link_adcs = []
             link_head = []
             # get all samples of a given link:
             for s in range(0,len(loc)):
                 link_adcs.append( get_adcdata_by_key(adcs[s],key) ) # basically get decoded data for e.g. evt['Prev2/Raw/Velo/DecodedADC'][1]
                 link_head.append( get_headerdata_by_key(headers[s],key) )

             for ll in range(0,len(linklist)):
                 l = linklist[ll]
                 for k in channellist: 
                     channeldata = [0.0] * len(loc)
                     for s in range(0,len(loc)): 
                         channeldata[s] = abs(link_adcs[s][l*32+k] - pedestals[l*32+k])
                     if sum(channeldata)>channeldatacut:
                        for s in range(0,len(loc)): 
                            pulshists[key][ll].Fill(xx[s],link_adcs[s][l*32+k])
                            timeslot = xx[s]
                            if timeslot > -50 and timeslot < -20:                                     # we see some strange entries in the area between -50 and -20 ns that would badly impact the average value for the pulsshape. 
                                if link_adcs[s][l*32+k] < 515:                                        # these "positive" values greater 515 need to be disregarded. 
                                    completepulsehists[key][l][timeslot].append(link_adcs[s][l*32+k]) 
                            else: completepulsehists[key][l][timeslot].append(link_adcs[s][l*32+k])   
#                               stephists[key][ll*nsteps+step].Fill(xx[s],link_adcs[s][l*32+k])
                  # # Get the header bits from ADC header data:
                  # for b in range(16):      # b = beetle index, 0 to 15
                  #       headbits = [0]*16  # to store bits of one full beetle  (4x4 bits = 16 bits)
                  #       for p in range(4): # port p = 0 to 3
                  #           l = b * 4 + p  # link l = b * 4 + p, 0 to 63
                  #          #pedestal = sum(pedestals[l*32:(l+1)*32])/32.0
                  #          #for k in range(4): headbits[k+4*p] = int(hdata[l*4+k] < pedestal) # header bit k=0,1,2,3 in port p 
                  #           for k in range(4): headbits[k+4*p] = int(hdata[l*4+k] < 512) # header bit k=0,1,2,3 in port p 
                  #       Beetpcn = Get_pcn_from_beetle_ports(headbits)
                  #       tell1_beepcns.append(Beetpcn)
                  #       if TAEfirstPCN == 0: 
                  #          TAEfirstPCN = Beetpcn
                  #       else: 
                  #          if Beetpcn != (TAEfirstPCN+s) : badpcn = True # at least one inconsistent PCN in the TAE Event

      else:
             skipped += 1


print 'pulseshape_histograms: got data over ',ncheck,' events out of ',n-1
for s in range(0,len(loc)):
    print '                     found ',countmissing[s],' events with missing data in',loc[s]+'Raw/Velo/... '
print 'Got in total ',n_badpcn,' bad PCN events (at least one inconsistent PCN in the TAE Event)'
print 'Got in total ',not_wanted_bcid,' TAEs with an unwanted BCID from ODIN'

for timeslot in completepulsehists[1][40].keys():
   for key in tell1numbers:
     for ll in range(0,len(linklist)):
       l = linklist[ll]
       averagepulshists[key][l].append(np.average(completepulsehists[key][l][timeslot]))



f.Write()
f.Close()
#---------------------
# save results to pickle for this run
#---------------------

pkl = pkldir+'Run_'+str(runnumber)+'_averagePulseshapePerLink.pkl.gz'
print "- Saving data to file",pkl

pkl_file = gzip.open(pkl, 'wb') # save as compressed gzip files directly
if not pkl_file:
    print "ERROR: could not open pickle file",pkl
    sys.exit()

pickle.dump(averagepulshists, pkl_file, pickle.HIGHEST_PROTOCOL)
pkl_file.close()

print 'done'

#return pulshists,stephists
#pulshists,stephists = pulseShapeHistograms(appMgr,nevtsperstep,nsteps,[1],bcidlist,peds,channeldatacut)





adcs    = evt['Raw/Velo/DecodedADC']
headers = evt['Raw/Velo/DecodedHeaders']
for sen in adcs.containedObjects():
   key = sen.key()
   #print '              sen key = ', key
   if key in tell1numbers:
      headerbitsfromdata_sample_tell1 = []
      sdata = sen.decodedData()
      hdata = get_headerdata_by_key(headers,key)
      



