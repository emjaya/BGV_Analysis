import matplotlib as mpl
#mpl.use('Agg')
#mpl.use('Agg')
mpl.interactive(True)
import numpy as np
from numpy import random,histogram,arange,sqrt,exp,nonzero
import scipy as sp
#import pylab as P
import matplotlib.pyplot as plt
import matplotlib.dates as md
import matplotlib.cm as cm
from matplotlib.dates import *
from scipy import optimize

# fname = myhist1480.txt
# f = open(filename,'r')

mylist = [[0, 0, 0], [1, 3, 5], [0, 2, 0], [0, 0, 0], [3, 4, 1], [0, 0, 0]]
fig = plt.figure(figsize=(16,7.5), facecolor = 'white')
ax = fig.add_subplot(111)
for i in range(len(mylist)):
	ax.plot(mylist[i])

# raws = [(np.random.rand(20, 100), np.random.rand(20, 100))]

# f, axes = plt.subplots(len(raws[0]), 1)

# for i in range(len(raws[0])):
# 	axes[i].plot(raws[0][i])
