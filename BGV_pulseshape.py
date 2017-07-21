#!/usr/bin/env python
# inspired from velo2-assembly.py
# Author: Karol Hennessy - karol.hennessy@cern.ch
# Author: Massimiliano Ferro-Luzzi
# Last edited: 26 March 2012

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
#f = TFile(rootfile, 'recreate')

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

def justcount_events(appMgr,nevts):

    print '###### Just count events '
    f = TFile('Justcount_run_'+runnumber+'.root', 'recreate')
    n = 0
    n_no_odin = 0
    nbcid  = [0] * 3564
    nbctyp = [0] * 4
    n_missingsample = 0
    odinbcid = TH1F('odin_bcid', 'odin_bcid', 3566, -1.5, 3564.5) 
    while (n<nevts):
          n += 1
          result = appMgr.run(1)
          #print 'Event ',n,' result = ',result
          bcid  =  -1
          if  not evt['/Event'].__nonzero__() : 
              print '###### Broken event ??? or end of files...  Event ',n
              break
          else:
              for s in range(0,len(loc)):
                  adcs    = evt[loc[s]+'Raw/Velo/DecodedADC']
                  headers = evt[loc[s]+'Raw/Velo/DecodedHeaders']
                  if not(adcs and headers):
                     print '   MISSING ADCS or HEADERS DATA in loc [',s,'] ',loc[s]+'Raw/Velo/... !!!'
                     n_missingsample += 1
              odinloc = 'DAQ/ODIN' #odinloc = 'Raw/Velo/EvtInfo'
              odin = evt[odinloc]
              if odin: 
                 bcid  =  odin.bunchId()
                 bctyp =  odin.bunchCrossingType()
                 orbnr =  odin.orbitNumber()
                 evtim = (odin.eventTime()).ns()
                 evtyp =  odin.eventType()
                 nbcid[bcid]  += 1
                 nbctyp[bctyp] += 1
              else: 
                 n_no_odin += 1
              odinbcid.Fill(bcid)

    print 'Looped over ',n,' events'
    print '        No Odin: ', n_no_odin
    print '        Missing sample: ', n_missingsample
    print 'Crossing types:'
    for i in range(4):    print nbctyp[i]
    print 'Counts per BCIDs:'
    for i in range(0,99): print nbcid[0+i*36:36+i*36]

    f.Write()
    f.Close()

    return odinbcid

def make_raw_histograms(appMgr,nevts,tell1numbers):
    print '###### Make raw histograms per link'
    f = TFile('Make_raw_run_'+runnumber+'.root', 'recreate')
    init = 0
    frawhists = {}
    for key in tell1numbers:
        frawhists[key] = []
        print 'Created frawhists[key=',key,'] list'
    count_Beetpcn_discrepancy = 0
    ncheck = 0
    n = 0
    nbcid  = [0] * 3564
    nbctyp = [0] * 4
    oldevtim = 0
    oldbcid = 0
    oldorbnr = 0
    while (n<nevts):
          n += 1
          skipevent = False
          result = appMgr.run(1)
          #print 'Event ',n,' result = ',result
          if  not evt['/Event'].__nonzero__() : 
              print '###### Broken event ???   Event ',n
              break
              #skipevent = True
          for s in range(0,len(loc)):
              adcs    = evt[loc[s]+'Raw/Velo/DecodedADC']
              headers = evt[loc[s]+'Raw/Velo/DecodedHeaders']
              if not(adcs and headers):
                 print '   MISSING ADCS or HEADERS DATA in loc [',s,'] ',loc[s]+'Raw/Velo/... !!!'
                 countmissing[s] += 1
                 skipevent = True
          odinloc = 'DAQ/ODIN' #odinloc = 'Raw/Velo/EvtInfo'
          odin = evt[odinloc]
          if odin : 
             bcid  =  odin.bunchId()
             bctyp =  odin.bunchCrossingType()
             orbnr =  odin.orbitNumber()
             evtim = (odin.eventTime()).ns()
             evtyp =  odin.eventType()
             nbcid[bcid]  += 1
             nbctyp[bctyp] += 1
             if evtim<oldevtim:
                #print 'WHAT THE HELL ? evtim = ',evtim,' < oldevtim = ',oldevtim
                #print '   ODIN OLD  bcid / orbnr / evtim :',oldbcid,oldorbnr,oldevtim
                #print '   ODIN THIS bcid / orbnr / evtim :',bcid,orbnr,evtim
                 told = ((oldevtim-1451606400)*1000-(oldorbnr*3564+oldbcid)*24950)/1000
                 tnew = ((evtim   -1451606400)*1000-(orbnr   *3564+bcid   )*24950)/1000
                #print '   ODIN OLD  (evtim-1451606400) - (orbnr*3564+bcid)*24.95 :',told 
                #print '   ODIN THIS (evtim-1451606400) - (orbnr*3564+bcid)*24.95 :',tnew 
                #print '                  difference OLD-NEW    :',told-tnew 
           # else: 
           #     print '                 '
             oldevtim = evtim
             oldbcid  = bcid
             oldorbnr = orbnr
          else:
             print 'No ODIN data in ',odinloc
          if not(skipevent): 
             #print 'Event ',n,'  bcid ',bcid,'  orbit number ',orbnr
             ncheck += 1
             headerbitsfromdata_allsamples = []
             for s in range(0,len(loc)):
                 adcs    = evt[loc[s]+'Raw/Velo/DecodedADC']
                 headers = evt[loc[s]+'Raw/Velo/DecodedHeaders']
                 evtinfo = evt[loc[s]+'Raw/Velo/EvtInfo']
                 #odin = evt[loc[s]+'DAQ/ODIN']
                 #if   odin:  print 'Sample ',s,' has ODIN bank'
                 #else     :  print 'Sample ',s,' has   NO  ODIN bank'
                 #print '   Sample loc[',s,'] ',loc[s]
                 #print '          adcs.size = ', adcs.size(), '  headers.size = ', headers.size() 
                 headerbitsfromdata_sample = []
                 for sen in adcs.containedObjects():
                     key = sen.key()
                     #print '              sen key = ', key
                     if key in tell1numbers:
                        headerbitsfromdata_sample_tell1 = []
                        sdata = sen.decodedData()
                        hdata = get_headerdata_by_key(headers,key)
                        #print '   key ',key,'       sdata.size = ',sdata.size()
                        #print '   key ',key,'       hdata.size = ',hdata.size()
                        if init == 0 :
                           for l in range(64):
                               hname = "h_raw_source%03d"%(key)+"_link%03d"%(l)+"_"+(loc[s].replace('/',''))
                               htitl = "raw data source "+str(key)+" link "+str(l)+"  "+(loc[s].replace('/',''))
                               frawhists[key].append( TH2F(hname, htitl, 36, -4.5, 31.5, 200, 412.5,611.5) )
                               print 'At index ',len(frawhists[key])-1,'   created histogram ',hname

                           init = 1 
                           for ki in tell1numbers: 
                               if len(frawhists[ki]) < (64*len(loc)): init = 0 
                           if init == 1: print 'Stop creating histograms'
                        for l in range(64): 
                              ll = 64*s+l
                              for k in range(32): 
                                  frawhists[key][ll].Fill(k,sdata[l*32+k])
                              for k in range( 4): 
                                  frawhists[key][ll].Fill(k-4,hdata[l*4+k])
                        #####################################################
                        """
                        #print 'TELL1 key ',key
                        bc_0,l0id_0,fempcn_0 = check_velo_pcn_event_info(evtinfo)
                          
                        # Get the header bits from ADC header data:
                        for b in range(16):      # b = beetle index, 0 to 15
                              headbits = [0]*16  # to store bits of one full beetle  (4x4 bits = 16 bits)
                              for p in range(4): # port p = 0 to 3
                                  l = b * 4 + p  # link l = b * 4 + p, 0 to 63
                                  for k in range(4): headbits[k+4*p] = int(hdata[l*4+k] < 512) # header bit k=0,1,2,3 in port p 
                              # finished one beetle
                              beetpcn,Beetpcn,R_beetpcn,R_Beetpcn = get_pcn_from_beetle_ports(headbits)
                              if (n%200==0) and (b==1 or b==8): 
                                 if s == len(loc)/2:
                                    print 'Evt ',n,' Sampl ',s,' Key ',key,' Bee ',b,' : ',headbits,' BCID ',bcid,bcid%187,' Orb ',\
                                          orbnr,(orbnr*3546+bcid)%187,' BeePCN ',Beetpcn,' TL1 (bc,l0,fempcn):',bc_0,l0id_0,fempcn_0
                                 else              :
                                    print 'Evt ',n,' Sampl ',s,' Key ',key,' Bee ',b,' : ',headbits,' BCID ','    ','    ',' Orb ',\
                                          '    ','    ',' BeePCN ',Beetpcn,' TL1 (bc,l0,fempcn):',bc_0,l0id_0,fempcn_0
                                #print  beetpcn,' --> ',Beetpcn,' <-- ',R_beetpcn,R_Beetpcn,255-beetpcn,255-Beetpcn,255-R_beetpcn,255-R_Beetpcn 
                              headerbitsfromdata_sample_tell1.append(headbits)

                        headerbitsfromdata_sample.append(headerbitsfromdata_sample_tell1)

                 headerbitsfromdata_allsamples.append(headerbitsfromdata_sample)

             # check consistency of header bits and odin bcid:
             for s in range(0,len(headerbitsfromdata_allsamples)):
                 port_firstbit = [1,0,1,0]
                 port_seconbit = [1,0,1,0]
                 current_Beetpcn = -1
                 Beetpcn_discrepancy = False
                 hbits_sample = headerbitsfromdata_allsamples[s]
                 for t in range(0,len(hbits_sample)):
                       hbits_sample_tell1 = hbits_sample[t]
                       for b in range(0,len(hbits_sample_tell1)): 
                           bee = hbits_sample_tell1[b]
                          #for port in range(4): 
                          #    if bee[4*port]!=port_firstbit[port]: print 'port ',port,' first bit changes --------------------'
                           Beetpcn = Get_pcn_from_beetle_ports(bee)
                           if current_Beetpcn == -1: current_Beetpcn = Beetpcn
                           if Beetpcn != current_Beetpcn: Beetpcn_discrepancy = True
                 if Beetpcn_discrepancy:
                    count_Beetpcn_discrepancy += 1
                    #print 'Beetpcn_discrepancy:  Event ',n,'  Sample ',s,'  BCID : ',bcid
                    #print hbits_sample

    print 'make_raw_histograms: got data over ',ncheck,' events out of ',n
    for s in range(0,len(loc)):
        print '                     found ',countmissing[s],' events with missing data in',loc[s]+'Raw/Velo/... '
    for i in range(0,4): print 'nbctyp[',i,'] = ',nbctyp[i]
    print 'Found ',count_Beetpcn_discrepancy,' count_Beetpcn_discrepancy'
    """

    f.Write()
    f.Close()

    return frawhists


def check_velo_pcn_event_info(tell1evtinfo):
    pcn_0    = tell1evtinfo.PCNBeetle(0)
    bc_0     = tell1evtinfo.bunchCounter(0)
    l0id_0   = tell1evtinfo.l0EventID(0)
    fempcn_0 = tell1evtinfo.FEMPCNG(0)
    st = ['','','','']
    unmatched = False
    st[0] = 'ERROR: PP %4i'%(0)+' : bc %4i'%bc_0+'  l0id %4i'%l0id_0+'  fempcn %4i'%fempcn_0+' beepcns '
    for b in range(4): st[0] += ' %4i'%pcn_0[b]
    for pp in range(1,4):
        pcn    = tell1evtinfo.PCNBeetle(pp)
        bc     = tell1evtinfo.bunchCounter(pp)
        l0id   = tell1evtinfo.l0EventID(pp)
        fempcn = tell1evtinfo.FEMPCNG(pp) 
        st[pp] = 'ERROR: PP %4i'%pp+' : bc %4i'%bc+'  l0id %4i'%l0id+'  fempcn %4i'%fempcn+' beepcns '
        if (bc != bc_0 or l0id != l0id_0 or fempcn != fempcn_0): unmatched = True
        for b in range(4): 
           st[pp] += ' %4i'%pcn[b]
           if (pcn[b] != pcn_0[b]): unmatched = True
    if unmatched :
        for pp in range(4): print st[pp] 

    return bc_0,l0id_0,fempcn_0

def get_pcn_from_beetle_ports(bee):
    # 12-15 = port 0, 8-11 = port 1, 4-7 = port 2, 0-3 = port 3      but without reversing header bits
    pcn = [bee[2],bee[3],bee[6],bee[7],bee[10],bee[11],bee[14],bee[15]] # MSB on the left  = pcn[0], LSB on the right = pcn[7]
    oth = [bee[0],bee[1],bee[4],bee[5],bee[8] ,bee[9] ,bee[12],bee[13]]
    # 12-15 = port 0, 8-11 = port 1, 4-7 = port 2, 0-3 = port 3      and reversing header bits pairwise
    R_pcn = ([bee[3],bee[2],bee[7],bee[6],bee[11],bee[10],bee[15],bee[14]])
    R_oth = ([bee[1],bee[0],bee[5],bee[4],bee[9],bee[8],bee[13],bee[12]])
    # 12-15 = port 3, 8-11 = port 2, 4-7 = port 1, 0-3 = port 0      but without reversing header bits
    Pcn = ([bee[14],bee[15],bee[10],bee[11],bee[6],bee[7],bee[2],bee[3]])
    Oth = ([bee[12],bee[13],bee[8],bee[9],bee[4],bee[5],bee[0],bee[1]])
    # 12-15 = port 3, 8-11 = port 2, 4-7 = port 1, 0-3 = port 0      and reversing header bits pairwise
    R_Pcn = [bee[15],bee[14],bee[11],bee[10],bee[7],bee[6],bee[3],bee[2]] # MSB on the left  = Pcn[0], LSB on the right = Pcn[7]
    R_Oth = [bee[13],bee[12],bee[9] ,bee[8] ,bee[5],bee[4],bee[1],bee[0]]
    bits,beetpcn   = binary2decimal(pcn)
    bits,Beetpcn   = binary2decimal(Pcn)   # SEEMS TO BE THE CORRECT ONE ???
    bits,R_beetpcn = binary2decimal(R_pcn)
    bits,R_Beetpcn = binary2decimal(R_Pcn) # OR THIS ONE ???
    return beetpcn,Beetpcn,R_beetpcn,R_Beetpcn

def Get_pcn_from_beetle_ports(bee):
    # 12-15 = port 3, 8-11 = port 2, 4-7 = port 1, 0-3 = port 0      but without reversing header bits
    Pcn = ([bee[14],bee[15],bee[10],bee[11],bee[6],bee[7],bee[2],bee[3]])
    Oth = ([bee[12],bee[13],bee[8],bee[9],bee[4],bee[5],bee[0],bee[1]])
    bits,Beetpcn   = binary2decimal(Pcn)   # SEEMS TO BE THE CORRECT ONE ???
    return Beetpcn

def binary2decimal(pcn):
    dec = 0
    bit = 0
    for m in range(0,8): dec += (2**(7-m)) * pcn[m]
    for m in range(0,8): bit += (10**(7-m)) * pcn[m]
    return bit,dec

def pulseShapeHistograms(appMgr,nevtsperstep,nsteps,tell1numbers,wanted_bcid_list,peds,channeldatacut):
    print '###### Make pulse shape histograms '
    f = TFile('PulseShape_run_'+runnumber+'.root', 'recreate')
    channellist = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
    #channellist = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
    linklist = [20,32,40]
    #linklist = range(64)
    pulshists = {}
    stephists = {}
    for key in tell1numbers:
        pulshists[key] = []
        stephists[key] = []
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
            for st in range(nsteps):
                hname = "h_PSscan_source%03d"%(key)+"_link%03d"%(l)+"_step%02d"%(st)
                htitl = "PulseShapeScan source "+str(key)+" link "+str(l)+" step "+str(st)
                stephists[key].append( TH2F(hname, htitl, nxbi, xmin,xmax, 150, 372.5,671.5) )
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
              a = evt[loc[s]+'Raw/Velo/DecodedADC']    
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
                     link_adcs.append( get_adcdata_by_key(adcs[s],key) )
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
                                stephists[key][ll*nsteps+step].Fill(xx[s],link_adcs[s][l*32+k])
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

    f.Write()
    f.Close()

    return pulshists,stephists

def view_single_event(appMgr,source,beetle,skipsomanyevents,peds):
    print '#### View single event'
    appMgr.run(skipsomanyevents)
    appMgr.run(1)
    odinloc = 'DAQ/ODIN' #odinloc = 'Raw/Velo/EvtInfo'
    odin = evt[odinloc]
    if odin : 
       bcid  =  odin.bunchId()
       bctyp =  odin.bunchCrossingType()
       orbnr =  odin.orbitNumber()
       evtim = (odin.eventTime()).ns()
       evtyp =  odin.eventType()
       print 'ODIN bcid = ', bcid,' orbnr = ',orbnr,' evtim = ',evtim,' evtyp = ',evtyp

    fig = plt.figure(figsize=(10,10))
    ax = []
 
    TAEfirstPCN = 0
    hea = np.linspace(0,7,8,dtype=np.int)
    hea = ((hea/2)*34 + hea%2)
    cha = np.linspace(0,127,128,dtype=np.int)
    cha = 2+((cha/32)*34 + cha%32)
    for s in range(0,len(loc)):
        ax.append(fig.add_subplot(len(loc),1,s+1))
        adcs    = evt[loc[s]+'Raw/Velo/DecodedADC']
        headers = evt[loc[s]+'Raw/Velo/DecodedHeaders']
        evtinfo = evt[loc[s]+'Raw/Velo/EvtInfo']
        print 'Sample loc[',s,'] ',loc[s]
        print '       adcs.size = ', adcs.size(), '  headers.size = ', headers.size() 
        for sen in adcs.containedObjects():
            key = sen.key()
            if key == source:
               bc_0,l0id_0,fempcn_0 = check_velo_pcn_event_info(evtinfo[key])
               print '              sen key = ', key
               print '              TELL1 BC = ',bc_0,' L0ID = ',l0id_0,' fempcn_0 = ',fempcn_0
               sdata = sen.decodedData()
               hdata = get_headerdata_by_key(headers,key)
               pedestals = peds[key]
               print '                     sdata.size = ',sdata.size()
               print '                     hdata.size = ',hdata.size()
               # Get the header bits from ADC header data:
               b  = beetle        # b = beetle index, 0 to 15
               dataadcs = [0]*128 # to store ADCs of one full beetle  
               dh       = [0]*16  # to store ADCs of header in one full beetle  
               headbits = [0]*16  # to store bits of one full beetle  (4x4 bits = 16 bits)
               pede     = pedestals[b*128:(b+1)*128]
               for p in range(4): # port p = 0 to 3
                   l = b * 4 + p  # link l = b * 4 + p, 0 to 63
                   pedepede   = pedestals[l*32:(l+1)*32]
                   avpedestal = sum(pedepede)/32.0
                   print 'Using avg pedestal ',avpedestal
                   for k in range(4):  dh      [k+4*p]  = hdata[l*4+k]            # header data k=0,1,2,3 in port p 
                   for k in range(4):  headbits[k+4*p]  = int(hdata[l*4+k] < avpedestal) # header bit k=0,1,2,3 in port p 
                  #for k in range(4):  headbits[k+4*p]  = int(hdata[l*4+k] < 512) # header bit k=0,1,2,3 in port p 
                   for k in range(32): 
                       dataadcs[k+32*p] = sdata[l*32+k]           # data k=0,...31       in port p 
               Beetpcn = Get_pcn_from_beetle_ports(headbits)
               print 'Bee ',b,' PCN ',Beetpcn
               if TAEfirstPCN == 0: 
                  TAEfirstPCN = Beetpcn
                  print 'First PCN: Bee ',b,' PCN ',TAEfirstPCN
               else: 
                  if Beetpcn != (TAEfirstPCN+s) : 
                     print 'bad PCN ',Beetpcn,' event (PCN in the TAE Event) Bee ',b,' TAEfirstPCN =',TAEfirstPCN
 
        ax[s].plot(cha,dataadcs,'ko')
        ax[s].plot(hea,[dh[14],dh[15],dh[10],dh[11],dh[6],dh[7],dh[2],dh[3]],'rs') 
        ax[s].plot(cha,pede,'.') 
        #drawhist(ax[s],hea,datahead)
        #drawhist(ax[s],cha,dataadcs)
    fig.savefig('single_event_L0id%i'%l0id_0+'_source%i'%source+'_beetle%i'%beetle+'.pdf', bbox_inches='tight')

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
    with open('/afs/cern.ch/user/m/mrihl/BGVAnalysis/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    #with open('obj/' + name + '.pkl', 'rb') as f:
    with open('/afs/cern.ch/user/m/mrihl/BGVAnalysis/'+ name + '.pkl', 'rb') as f:
        return pickle.load(f)

############################ main: ############################################################

skipevts = 0
if skipevts > 0:
    print '###### Skip ', skipevts, ' events'
    appMgr.run(skipevts)
#frawhists = make_raw_histograms(appMgr,nevts,[1])


#odinbcid=justcount_events(appMgr,nevts)
#bgvtell1numbers=[1,2,4,5,6,7,8,9] # 2015: nr 9 is not receiving data from FE
bgvtell1numbers=[1]  
#frawhists = make_raw_histograms(appMgr,nevts,[1])
#frawhists = make_raw_histograms(appMgr,nevts,[1,2,4,5,6,7,8,9])
#frawhists = make_raw_histograms(appMgr,nevts,bgvtell1numbers,False)

nevtsperstep = 1000
nsteps = 25
#do_peds = False
do_peds = True
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
 # ---->  peds = load_obj('peds_'+runnumber+'_v1')
   appMgr.run(nevtsperstep) # skip first step!
# ---->for key in bgvtell1numbers: 
# ---->    print 'peds[',key,'][0:5] = ',peds[key][0:5],' ...'

channeldatacut = 50
pulshists,stephists = pulseShapeHistograms(appMgr,nevtsperstep,nsteps,[1],bcidlist,peds,channeldatacut)

#sum_h4pos     = {}
#sum_h4neg     = {}
#nor_h4pos     = {}
#nor_h4neg     = {}

#                    sum_h4pos[key] = [0.0]*2048
#                    sum_h4neg[key] = [0.0]*2048
#                    nor_h4pos[key] = [0.0]*2048
#                    nor_h4neg[key] = [0.0]*2048
#print '###### Normalize and plot correlation check vectors'
#sizex = 40
#sizey = 40
#fig  = plt.figure(figsize=(sizex,sizey))
#fig.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
#cha = np.linspace(0,2047,2048)
#n = 0
#for key in sum_h4pos:
#    h4pos = [(x/float(y+1e-30)) for x,y in zip(sum_h4pos[key],nor_h4pos[key])]
#    h4neg = [(x/float(y+1e-30)) for x,y in zip(sum_h4neg[key],nor_h4neg[key])]
#    h4dif = [(x-y) for x,y in zip(h4pos,h4neg)]
#    ax = fig.add_subplot(2,4,n+1)
#    ax.set_title('Source '+str(key))
#    ax.set_xlabel('channel')
#    ax.set_ylabel('pedsub mean for h4>512, h4<512 and diff')
#    ax.plot(cha,h4pos,'k.')
#    ax.plot(cha,h4neg,'r.')
#    ax.plot(cha,h4dif,'g.')
#    ax.set_xlim(-1.0,2048.0)
#    ax.set_ylim(-10.0,10.0)
#    n += 1
#name='headcrosscheck.pdf'
#fig.savefig(name, bbox_inches='tight')

        
