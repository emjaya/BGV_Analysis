#!/usr/bin/env python
# Author: Karol Hennessy - karol.hennessy@cern.ch
# Last edited: 26 March 2012

from Gaudi.Configuration  import *
from GaudiPython.Bindings import AppMgr
from GaudiPython.Bindings import gbl
from Configurables        import LHCbApp
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
nevts =1000 
skipevts=1

EventSelector().PrintFreq = 100
#datafile              = os.environ['VELO2DATAFILE']
datafile              = sys.argv[1] #input file has to be given as argument when velo2-assembly.py is called
rootfile              = datafile.replace('.mdf', '_v2a.root')
#rootfile              = 'test.root' 
EventSelector().Input = [ "DATAFILE='"+datafile+"' SVC='LHCb::MDFSelector'" ]
#EventSelector().Input = [ "DATAFILE = 'file:/calib/velo/assembly/velo2/mod102-vac-m30-20120125-104723.mdf' SVC = 'LHCb::MDFSelector'" ]

# init
appMgr      = AppMgr()
appMgr.addAlgorithm('LbAppInit')
appMgr.addAlgorithm('PrepareVeloFullRawBuffer')
appMgr.addAlgorithm('DecodeVeloFullRawBuffer')
#appMgr.addAlgorithm('nzsStreamListener')


evt         = appMgr.evtsvc()
#data
#evtsel      = appMgr.evtsel()
#evtsel.PrintFreq = 100
#evtsel.Input = [ "DATAFILE='"+datafile+"' SVC='LHCb::MDFSelector'" ]
#evtsel.open([ 'PFN:'+datafile ])
f = TFile(rootfile, 'recreate')

# algs
prep              = appMgr.algorithm('PrepareVeloFullRawBuffer')
decode            = appMgr.algorithm('DecodeVeloFullRawBuffer')
prep.RunWithODIN  = False
decode.CableOrder = [3,2,1,0]


# event loop

#print appMgr.algorithms()

# hists
headhists     = {}
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
                if i == 0:
                        peds[key] = [0.0]*2048
                peds[key] = [ (x+y) for x,y in zip(peds[key], sen.decodedData())]
        headers = evt['Raw/Velo/DecodedHeaders']
        for h in headers.containedObjects():
                key = h.key()
                if i == 0:
                        headhists[key] = TH2F("h_header_%03d" % (key), 'headers in sensor '+str(key), 256, -0.5, 255.5, 200, 411.5, 611.5)

                hdata = h.decodedData()
                for j, d in enumerate(hdata):
                        headhists[key].Fill(j, d)

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
        adcs = evt['Raw/Velo/DecodedADC'] # location in TES (transient event store)
        for sen in adcs.containedObjects():
                key = sen.key() # gives number of Tell1 Sensor
                raw = sen.decodedData() # should come from VeloTELL1Data function m_decodedData
                pedsub  = [ (raw[j] - peds[key][j]) for j in range(2048)]
                corr = mcms(pedsub)
                # noise
                if i == 0 :
                        sum_vals[key] = [0.0]*2048
                        sum_vals2[key] = [0.0]*2048
                        sum_vals_raw[key] = [0.0]*2048
                        sum_vals2_raw[key] = [0.0]*2048
                        pedsubhists[key] = TH2F("h_pedsub_adcs_%03d" % (key), 'pedestal subbed ADCs in sensor '+str(key), 2048, -0.5, 2047.5, 200, -100.5, 99.5)
                for j, d in enumerate(pedsub):
                        pedsubhists[key].Fill(j, d)

                sum_vals[key] = [ (x+y) for x,y in zip(sum_vals[key], corr)]
                sum_vals2[key] = [ (x+(y*y)) for x,y in zip(sum_vals2[key], corr)]
                sum_vals_raw[key] = [ (x+y) for x,y in zip(sum_vals[key], pedsub)]
                sum_vals2_raw[key] = [ (x+(y*y)) for x,y in zip(sum_vals2[key], pedsub)]

#noise
noise = {}
rawnoise = {}
for key in sum_vals:
        noise[key] = [ sqrt((y)/float(nevts)) for x,y in zip(sum_vals[key], sum_vals2[key])]
        rawnoise[key] = [ sqrt((y)/float(nevts)) for x,y in zip(sum_vals_raw[key], sum_vals2_raw[key])]
        noisehists[key] = TH1F("h_noise_%03d" % (key), 'noise in sensor '+str(key), 2048, -0.5, 2047.5)
        rawnoisehists[key] = TH1F("h_rawnoise_%03d" % (key), 'noise in sensor '+str(key), 2048, -0.5, 2047.5)
        for j in range(2048) :  
                noisehists[key].SetBinContent(j+1, noise[key][j])
                rawnoisehists[key].SetBinContent(j+1, rawnoise[key][j])

f.Write()
f.Close()
        

