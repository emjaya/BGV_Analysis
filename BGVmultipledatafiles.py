#!/usr/bin/env python
# Author: Karol Hennessy - karol.hennessy@cern.ch
# Last edited: 26 March 2012

from Gaudi.Configuration  import *
from GaudiPython.Bindings import AppMgr
from GaudiPython.Bindings import gbl
from Configurables        import LHCbApp
from Configurables        import PrepareVeloFullRawBuffer
from Configurables        import DecodeVeloFullRawBuffer

# from GaudiKernel.ProcessJobOptions import importOptions
# importOptions("$VETRAROOT/options/Velo/TAEDecodingNZS-PulseShapeScan.py")

#from Configurables       import PrepareVeloFullRawBuffer
#from Configurables       import DecodeVeloFullRawBuffer
#from Configurables       import nzsStreamListener
#from Configurables       import LbAppInit
#from Configurables       import Vetra
#from Configurables       import (CondDB, CondDBAccessSvc)
import GaudiPython
from ROOT                 import TH2F, TCanvas, TFile, TH1F
import sys
from math import sqrt

lhcb=LHCbApp()
nevts = 50
skipevts=10

EventSelector().PrintFreq = 500
#datafile              = os.environ['VELO2DATAFILE']
#datafile              = sys.argv[1] #input file has to be given as argument when velo2-assembly.py is called
a = []

a.append("DATAFILE='file:///afs/cern.ch/work/m/mrihl/public/Run_0001589_20160427-102746.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
# a.append("DATAFILE='file:///afs/cern.ch/work/m/mrihl/public/Run_0001589_20160427-102853.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
# a.append("DATAFILE='file:///afs/cern.ch/work/m/mrihl/public/Run_0001589_20160427-102957.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
# a.append("DATAFILE='file:///afs/cern.ch/work/m/mrihl/public/Run_0001589_20160427-103101.bgvctrl.mdf' SVC='LHCb::MDFSelector'")
rootfile              = 'PulseShape2016_1589_plotter_headerRemoved27042016.root'
#rootfile              = 'test.root' 
EventSelector().Input = a
#EventSelector().Input = [ "DATAFILE = 'file:/calib/velo/assembly/velo2/mod102-vac-m30-20120125-104723.mdf' SVC = 'LHCb::MDFSelector'" ]
#loc = [ 'Prev3/'      , 'Prev2/'      , 'Prev1/'      , ''              , 'Next1/'      , 'Next2/'      , 'Next3/'        ]
loc      = [ 'Prev2/'      , 'Prev1/'      , ''              , 'Next1/'      , 'Next2/']
prepList = [ 'preparePrev2', 'preparePrev1', 'prepareCentral', 'prepareNext1', 'prepareNext2']
decoList = [ 'decodePrev2' , 'decodePrev1' , 'decodeCentral' , 'decodeNext1' , 'decodeNext2' ]
# init
appMgr      = AppMgr()
appMgr.addAlgorithm('LbAppInit')
#appMgr.addAlgorithm('PrepareVeloFullRawBuffer')
#appMgr.addAlgorithm('DecodeVeloFullRawBuffer')

appMgr.addAlgorithm('PrepareVeloFullRawBuffer')
appMgr.addAlgorithm('DecodeVeloFullRawBuffer')

for i in range(1,2+1): # TAE==2
        appMgr.addAlgorithm('PrepareVeloFullRawBuffer/preparePrev'+str(i))
        appMgr.addAlgorithm('DecodeVeloFullRawBuffer/decodePrev'+str(i))
        appMgr.addAlgorithm('PrepareVeloFullRawBuffer/prepareNext'+str(i))
        appMgr.addAlgorithm('DecodeVeloFullRawBuffer/decodeNext'+str(i))
        print 'PrepareVeloFullRawBuffer/preparePrev'+str(i)
        print 'DecodeVeloFullRawBuffer/decodePrev'+str(i)
        print 'PrepareVeloFullRawBuffer/prepareNext'+str(i)
        print 'DecodeVeloFullRawBuffer/decodeNext'+str(i)


#appMgr.addAlgorithm('nzsStreamListener')


evt               = appMgr.evtsvc()
#data
#evtsel      = appMgr.evtsel()
#evtsel.PrintFreq = 100
#evtsel.Input = [ "DATAFILE='"+datafile+"' SVC='LHCb::MDFSelector'" ]
#evtsel.open([ 'PFN:'+datafile ])
f = TFile(rootfile, 'recreate')

# algs
prep              = []
deco              = []

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

# prep.append(appMgr.algorithm('PrepareVeloFullRawBuffer'))
# decode.append(appMgr.algorithm('DecodeVeloFullRawBuffer'))
# for i in range(len(TAE)):
#         if i == 2: continue
#         prep.append(appMgr.algorithm(TAE[i]))
#         decode.append(appMgr.algorithm(TAE[i]))

# prep[0].RunWithODIN  = False
# decode[0].CableOrder = [3,2,1,0]

#for i in range(len(TAE)):
#        prep.append(appMgr.algorithm(TAE[i]))
#        decode.append(appMgr.algorithm(TAE[i]))

 #       prep[i].RunWithODIN  = False
 #       decode[i].CableOrder = [3,2,1,0]




# event loop

#print appMgr.algorithms()

# hists
headhists     = {}
headdata      = {1: [], 2: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

noisehists    = {}
rawnoisehists = {}
pedsubhists   = {}
pedhists      = {}
# arrays
peds          = {}
sum_vals      = {}
sum_vals2     = {}
sum_vals_raw  = {}
sum_vals2_raw = {}
ADCCenter = {1: [], 2: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
ADCPrev2 = {1: [], 2: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
ADCPrev1 = {1: [], 2: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
ADCNext1 = {1: [], 2: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
ADCNext2 = {1: [], 2: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

def mcms(data):
        corr = [0]*2048
        for i in range(64):
                x = 0
                for j in range(32):
                        ch = i*32+j
                        x += data[ch]
                x /= 32
                for j in range(32):
                        ch = i*32+j
                        corr[ch] = data[ch] - x
        return corr

def rewind(appMgr):
        algs = appMgr.algorithms()
        for i in algs:
                appMgr.algorithm(i).Enable = False
        appMgr.evtsel().rewind()
        for i in algs:
                appMgr.algorithm(i).Enable = True

if skipevts > 0:
        appMgr.run(skipevts)
for i in range(nevts):
        x=appMgr.run(1)
        if not evt['/Event'].__nonzero__() : break
        adcs = evt['Raw/Velo/DecodedADC']
        for sen in adcs.containedObjects():
                key = sen.key()
                #headdata[key] = []
                if i == 0:
                        peds[key] = [0.0]*2048
                peds[key] = [ (x+y) for x,y in zip(peds[key], sen.decodedData())]
        headers = evt['Raw/Velo/DecodedHeaders']
        for h in headers.containedObjects():
                key = h.key()
                if i == 0:
                        headhists[key] = TH2F("h_header_%03d" % (key), 'headers in sensor '+str(key), 256, -0.5, 255.5, 200, 411.5, 611.5)
                a = []
                hdata = h.decodedData()
                for j, d in enumerate(hdata):
                        headhists[key].Fill(j, d)
                        a.append(d)
                headdata[key].append(a)

                

#ped
for k,v in peds.iteritems():
        peds[k] = [(x/nevts) for x in v]
        pedhists[k] = TH1F("h_pedestal_%03d" % (k), 'pedestals in sensor '+str(k), 2048, -0.5, 2047.5)
        
        for ch in range(2048):
                pedhists[k].Fill(ch, peds[k][ch])

#rewind(appMgr)

for i in range(nevts):
        appMgr.run(1)
        if not evt['/Event'].__nonzero__() : break
        adcs  = evt['Raw/Velo/DecodedADC'] # location in TES (transient event store)
        prev2 = evt['Prev2/Raw/Velo/DecodedADC'] 
        prev1 = evt['Prev1/Raw/Velo/DecodedADC']
        next1 = evt['Next1/Raw/Velo/DecodedADC']
        next2 = evt['Next2/Raw/Velo/DecodedADC']
        for sen in adcs.containedObjects():
                key = sen.key() # gives number of Tell1 Sensor
                raw = sen.decodedData() # should come from VeloTELL1Data function m_decodedData
                pedsub  = [ (raw[j] - peds[key][j]) for j in range(2048)]
                adccenter = [ (raw[j] - peds[key][j] + peds[key][j] ) for j in range(2048)]
                corr = mcms(pedsub)
                # noise
                if i == 0 :
                        sum_vals[key] = [0.0]*2048
                        sum_vals2[key] = [0.0]*2048
                        sum_vals_raw[key] = [0.0]*2048
                        sum_vals2_raw[key] = [0.0]*2048
                        pedsubhists[key] = TH2F("h_pedsub_adcs_%03d" % (key), 'pedestal subbed ADCs in sensor '+str(key), 2048, -0.5, 2047.5, 200, -100.5, 99.5)
                adccent = []
                for j, d in enumerate(pedsub):
                        pedsubhists[key].Fill(j, d)
                #         pedsubhists[key] = []
                #         pedsubhists[key].append(d)
                for u, l in enumerate(adccenter):
                        adccent.append(l)
                ADCCenter[key].append(adccent)


                sum_vals[key] = [ (x+y) for x,y in zip(sum_vals[key], corr)] # gets list pair of sum_vals and corr
                sum_vals2[key] = [ (x+(y*y)) for x,y in zip(sum_vals2[key], corr)]
                sum_vals_raw[key] = [ (x+y) for x,y in zip(sum_vals[key], pedsub)]
                sum_vals2_raw[key] = [ (x+(y*y)) for x,y in zip(sum_vals2[key], pedsub)]

        for sen in prev2.containedObjects():
                key = sen.key() # gives number of Tell1 Sensor
                raw = sen.decodedData() # should come from VeloTELL1Data function m_decodedData
                adcprev2 = [ (raw[j] ) for j in range(2048)]
                                
                aprev2 = []
                for u, l in enumerate(adcprev2):
                        aprev2.append(l)
                ADCPrev2[key].append(aprev2)

        for sen in prev1.containedObjects():
                key = sen.key() # gives number of Tell1 Sensor
                raw = sen.decodedData() # should come from VeloTELL1Data function m_decodedData
                adcprev1 = [ (raw[j] ) for j in range(2048)]
                                
                aprev1 = []
                for u, l in enumerate(adcprev1):
                        aprev1.append(l)
                ADCPrev1[key].append(aprev1)

        for sen in next1.containedObjects():
                key = sen.key() # gives number of Tell1 Sensor
                raw = sen.decodedData() # should come from VeloTELL1Data function m_decodedData
                adcnext1 = [ (raw[j] ) for j in range(2048)]
                                
                anext1 = []
                for u, l in enumerate(adcnext1):
                        anext1.append(l)
                ADCNext1[key].append(anext1)

        for sen in next2.containedObjects():
                key = sen.key() # gives number of Tell1 Sensor
                raw = sen.decodedData() # should come from VeloTELL1Data function m_decodedData
                adcnext2 = [ (raw[j] ) for j in range(2048)]
                                
                anext2 = []
                for u, l in enumerate(adcnext2):
                        anext2.append(l)
                ADCNext2[key].append(anext2)

Center = {1: [], 2: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

for link in range(64): 
        chan = 32+link*32
        avglink = []
        for event in range(50): 
                centerlink = 0
                for channel in range(link*32, chan):
                
                        centerlink += (ADCCenter[1][event][channel])
                avglink.append(centerlink/32)
        Center[1].append(avglink) 





f.Write()
f.Close()
        

