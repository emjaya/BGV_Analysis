import matplotlib as mpl
#mpl.use('Agg')
#mpl.use('Agg')
#mpl.interactive(True)
import numpy as np
from numpy import random,histogram,arange,sqrt,exp,nonzero
np.set_printoptions(threshold='nan') # print whole array, not just first and last 3 entries
import scipy as sp
#import pylab as P
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.cm as cm
from matplotlib.dates import *
from scipy import optimize
from numpy.polynomial import Polynomial
#from mytools import *
#from classData import *
#from classDeconvolution import *
#import classResolution as res
import sys
import bz2, gc, marshal, gzip
from time import clock
try:
    import cPickle as pickle
except:
    import pickle


print 'Starting pulseShapeScanAnalyzer.py'
print 'You must have run bgv_raw.py first! '
if len(sys.argv)<2 :
   print 'You must provide the run number: python -i pulseShapeScanAnalyzer.py 1589 ' 
   sys.exit(-1)
for i in range(0,len(sys.argv)):
   print 'sys.argv[',i,'] = ',sys.argv[i]
runnumber              = sys.argv[1]

print runnumber
pkldir = '/afs/cern.ch/user/m/mrihl/BGVAnalysis/'

# read (remaining) LDM pickle
fname = pkldir+'Run_'+str(runnumber)+'_averagePulseshapePerLink.pkl.gz'
print ' - read from pickle file',fname
pkl_file = gzip.open(fname, 'rb')
# try:
#     pkl_file = gzip.open(fname, 'rb')
# except:
#     #fname = datadir+'BgStd_BG_'+str(fill)+'_'+str(run)+'_0.pkl.bz2'
#     pkl_file = bz2.BZ2File(fname, 'rb')

averagePulseShapePerLink = pickle.load(pkl_file)

pkl_file.close()

#-----------------------
# plot design parameters
#-----------------------

markers = {0:'s',1:'o',2:'^',3:'v', 4: '^', 5: 'd', 6: 'D'}
col = ['g','b','r','g','orange']
colors = {0:'g',1:'b',2:'r',3:'c', 4: 'deeppink', 5: 'BlueViolet', 6: 'DarkOrange'}
colors3 = {0:'g',1:'lawngreen',2:'mediumpurple',3:'k'}
colors4 = {0:'g',1:'OrangeRed',2:'gold',3:'k'}

mfc = {'X':'0','Y':'1','Z':'0.5'} # markerfacecolor
colors2 = {'X':'g','Y':'r','Z':'c'}
col = ['g','b','r','g','y']
colourmap = [cm.BuPu, cm.Greens, cm.BuPu, cm.GnBu, cm.PuRd, cm.PuBu, cm.YlGn, cm.YlGnBu, cm.BuGn, cm.BuPu]

prop = mpl.font_manager.FontProperties(size='small')
tell1list = [1, 2, 4, 5, 6, 7, 8, 9]
#tell1list = [1]
delayvalueperModule = [0]
avgPulseShapePerTell1 = {}
######## plot pulseshapes in different ways ##################
for tell1 in tell1list:
	avgPulseShapePerTell1[tell1] = []
	fig = plt.figure(figsize=(12,5.5), facecolor = 'white')
	ax = fig.add_subplot(111)
	ax.set_ylabel('average ADC counts per link')
	ax.set_xlabel('delay [ns]')
	ax.set_xlim(-63,62)
	ax.set_title('PulseShapes for links 0-63 for bgvtell0'+str(tell1))
	color=iter(colourmap[tell1](np.linspace(0,1,len(averagePulseShapePerLink[1][0])+5)))

######## plot all links per Tell1 in one plot ##################
	for link in range(len(averagePulseShapePerLink[1])): 
		#colour = str(0.01+1./65*(link+1))
		colour=next(color)
		ax.plot(
		      [x-62.5 for x in range(len(averagePulseShapePerLink[1][link]))]
		      ,averagePulseShapePerLink[tell1][link]
		      ,color = colour
		      ,linestyle='-'
		      ,label='PulseShape for bgvtell0'+str(1)+' link'+str(link)
		      )
		#plt.show()
		fig.savefig('PulseShapeScanPlots/Run_'+str(runnumber)+'_PulseShapes for bgvtell0'+str(tell1)+'.png',bbox_inches='tight')
	print 'saving plot for bgvtell0'+str(tell1)+' all links!'
	
######## plot 4 links per Beetle ##################	
	# color=iter(colourmap[tell1](np.linspace(0,1,len(averagePulseShapePerLink[1][0])+5)))
	# beetlenumber = 16
	# for beetle in range(16):
	# 	beetlenumber-=1
	# 	links = []
	# 	fig = plt.figure(figsize=(12,5.5), facecolor = 'white')
	# 	ax = fig.add_subplot(111)
	# 	ax.set_ylabel('average ADC counts per link')
	# 	ax.set_xlabel('delay [ns]')
	# 	ax.set_xlim(-63,62)
	# 	for link in range(4):
	# 		colour=next(color)
	# 		links.append(beetle*4+link)
	# 		ax.plot(
	# 		      [x-62.5 for x in range(len(averagePulseShapePerLink[1][link]))]
	# 		      ,averagePulseShapePerLink[tell1][beetle*4+link]
	# 		      ,color = colour
	# 		      ,linestyle='-'
	# 		      ,label='PulseShape for bgvtell0'+str(tell1)+' link'+str(beetle*4+link)
	# 		      )
	# 	ax.set_title('PulseShapes for bgvtell0'+str(tell1)+', Beetle '+str(beetlenumber)+', links '+str(links))
	# 	fig.savefig('PulseShapeScanPlots/Run_'+str(runnumber)+'_PulseShapes for bgvtell0'+str(tell1)+'_Beetle_'+str(beetlenumber)+'.png',bbox_inches='tight')
	# 	print 'saving plot for bgvtell0'+str(tell1)+' Beetle '+str(beetlenumber)+'!'

######## get average pulseshape for 64 links ##################	
	
	for timeslot in range(len(averagePulseShapePerLink[1][0])): # 125 slots
		valuepertimeslot = []
		for link in range(len(averagePulseShapePerLink[1])):
			valuepertimeslot.append(averagePulseShapePerLink[tell1][link][timeslot])
		avgPulseShapePerTell1[tell1].append(np.average(valuepertimeslot))
		if len(avgPulseShapePerTell1[tell1])==125:
			fig = plt.figure(figsize=(12,5.5), facecolor = 'white')
			ax = fig.add_subplot(111)
			ax.set_ylabel('average ADC counts per link')
			ax.set_xlabel('delay [ns]')
			ax.set_title('Average PulseShape for bgvtell0'+str(tell1))
			ax.set_xlim(-63,62)
			ax.plot(
			      [x-62.5 for x in range(len(averagePulseShapePerLink[1][link]))]
			      ,avgPulseShapePerTell1[tell1]
			      ,color = colour
			      ,linestyle='-'
			      ,label='Average PulseShape for bgvtell0'+str(1)
			      )
			fig.savefig('PulseShapeScanPlots/Run_'+str(runnumber)+'_average_PulseShapes for bgvtell0'+str(tell1)+'_all_links.png',bbox_inches='tight')
	#signalminimum = np.where(avgPulseShapePerTell1[tell1] == min(avgPulseShapePerTell1[tell1]))
	index_signalminimum = np.argmin(avgPulseShapePerTell1[tell1]) #nicer solution
	delayvalueperModule.append(62-index_signalminimum)

######## fit parameters ##################	
	
	xvalues = np.asarray([t-62.5 for t in range(len(averagePulseShapePerLink[1][link]))])
	if runnumber < '1700':
		idx = np.where((xvalues>-45) & (xvalues<-10))
	else: 
		idx = np.where((xvalues>-10) & (xvalues<27))
	x = xvalues[idx[0][0]:idx[0][-1]+1]
	y = avgPulseShapePerTell1[tell1][idx[0][0]:idx[0][-1]+1]
	#z = np.polyfit(x, y, 2)
	# p = np.poly1d(z)
	# p4 = np.poly1d(np.polyfit(x, y, 4))
	# p30 = np.poly1d(np.polyfit(x, y, 30))
	# xp = np.linspace(-10, 25, 100)
	# _ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p4(xp), '--')
	# plt.ylim(480,530)
	p = Polynomial.fit(x, y, 2)
	fig = plt.figure(figsize=(12,5.5), facecolor = 'white')
	ax = fig.add_subplot(111)
	ax.set_ylabel('average ADC counts per link')
	ax.set_xlabel('delay [ns]')
	ax.set_title('Average PulseShape for bgvtell0'+str(tell1)+' plus fit')
	ax.set_xlim(-63,62)
	ax.plot(xvalues
		, avgPulseShapePerTell1[tell1]
		, color = 'green'
		, marker = 'o'
		)
	ax.plot(*p.linspace())
	fig.savefig('PulseShapeScanPlots/Run_'+str(runnumber)+'_average_PulseShapes for bgvtell0'+str(tell1)+'_fit.png',bbox_inches='tight')


delayvalueperModule.insert(3, 0) #writes value zero for bgvtell03, since we don't have it
for tell1 in tell1list:
	print 'delay value for bgvtell0'+str(tell1)+': ', delayvalueperModule[tell1]

result = open('Run_'+str(runnumber)+'_PulseShapeScanResults.txt', 'w')
result.write('\n')
result.write('Run: %s \n' %(runnumber))
for tell1 in tell1list:
  result.write('delay value for bgvtell0%s: %s \n' %(tell1, delayvalueperModule[tell1]))
result.close()

