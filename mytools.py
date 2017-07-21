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

# tools for plots
prop = mpl.font_manager.FontProperties(size='small')

errampl = 1.

##########################
### functions
##########################
gnorm = lambda sigma: 1/(np.sqrt(2*np.pi)*abs(sigma))

# normalizationfactor for f*g=fg*factor
# p = [mu1,mu2,sigma1,sigma2]
gnormp = lambda p:sp.exp(-(p[0]-p[1])**2/(2*(p[2]**2+p[3]**2)))/np.sqrt(2*np.pi*(p[2]**2+p[3]**2))
# product of two Gaussians is again a Gaussian
# p = [sigma1,sigma2]
# p = [mu1,mu2,sigma1,sigma2]
gm_sigma = lambda p: p[0]*p[1]/np.sqrt(p[0]**2 + p[1]**2) 
gm_mu = lambda p: (p[1]*p[2]**2 + p[0]*p[3]**2)/(p[2]**2 + p[3]**2)

#xi = lambda b, N0, a: (a*N0)/b
xi = lambda tb, ta: ta/tb

mconst = lambda p, x: p * x
mconstResid = lambda p, x, y: (mconst(p,x)-y)
mconstResidErr = lambda p, x, y, yerr: mconstResid(p,x,y)/yerr

mline = lambda p, x: p[0] * x + p[1]
mlineResid = lambda p, x, y: (mline(p,x)-y)
mlineResidErr = lambda p, x, y, yerr: mlineResid(p,x,y)/yerr

mdecay_ = lambda x, p: p[0] * sp.exp(-p[1]*x)
mdecay = lambda p, x: p[0] * sp.exp(-p[1]*x)
mdecayResid = lambda p, x, y: (mdecay(p,x)-y)
mdecayResidErr = lambda p, x, y, yerr: mdecayResid(p,x,y)/yerr

ldecay = lambda p, x: p[0] * sp.exp(-2*(1/p[1])*x)
ldecayResid = lambda p, x, y: (ldecay(p,x)-y)
ldecayResidErr = lambda p, x, y, yerr: ldecayResid(p,x,y)/yerr

convol = lambda p: np.sqrt(p[0]**2 + p[1]**2)
convolResid = lambda p, x: (convol(p)-x)
convolResidErr = lambda p, x, xerr: convolResid(p,x)/xerr

# resolution parametrisation p=[A,B,C]
# x = number of tracks
# y = resolution
sigmares = lambda p, x: (p[0]/x**p[1])+p[2]
#sigmares = lambda p, x: (p[0]/x**p[1])+abs(p[2])
sigmaresResid = lambda p, x, y: (sigmares(p,x)-y)
sigmaresResidErr = lambda p, x, y, yerr: sigmaresResid(p,x,y)/yerr

# resolution parametrisation p=[A,B,C,D]
# x = number of tracks
# y = resolution
# a+bx+cx2 +dx3 =y
sigmaresd = lambda p, x: p[0] + (p[1]*np.power(x,1)) + (p[2]*np.power(x,2)) + (p[3]*np.power(x,3))
#sigmaresd = lambda p, x: (p[0]/x**p[1])+(p[2]*np.power(x-p[4],2)) + p[3]
sigmaresdResid = lambda p, x, y: (sigmaresd(p,x)-y)
sigmaresdResidErr = lambda p, x, y, yerr: sigmaresdResid(p,x,y)/yerr

ldecayxi = lambda p, x: p[0] / ( ( (1+(p[2]/p[1]) * sp.exp((1/p[1])*x) ) - (p[2]/p[1]) )**2 )
ldecayxiResid = lambda p, x, y: (ldecayxi(p,x)-y)
ldecayxiResidErr = lambda p, x, y, yerr: ldecayxiResid(p,x,y)/yerr

moffset = lambda p, x, slope: slope * x + p[0]
moffsetResid = lambda p, x, y, slope: (moffset(p,x,slope)-y)
moffsetResidErr = lambda p, x, y, yerr, slope: moffsetResid(p,x,y,slope)/yerr

# Z and betastar in mm
# sigma = sigmastar * sqrt(1 + z^2/betastar^2) (corr factor > 1)
def beamWidthBetaCorrected(z,betastar,sigma=1.):
    return sigma * np.sqrt(1 + (z**2/betastar**2))
def beamWidthBetaCorrectedResid(z,y,betastar,sigma=1.):
    return beamWidthBetaCorrected(z,betastar,sigma) - y
def beamWidthBetaCorrectedResidErr(z,y,yerr,betastar,sigma=1.):
    return beamWidthBetaCorrectedResid(z,y,betastar,sigma)/yerr

# p = [A,mu,sigma]
def mgaussi(x,p,muoffset=[]):
    of = 0.
    if len(muoffset)>1:
        of = muoffset[0]
    #print p,of
    return p[0]*gnorm(p[2])*sp.exp(-(x-(p[1]+of))**2/(2.0*p[2]**2))

mgauss = lambda p, x: p[0]*gnorm(p[2])*sp.exp(-(x-p[1])**2/(2.0*p[2]**2))
mgaussResid = lambda p, x, y: (mgauss(p,x)-y)
mgaussResidErr = lambda p, x, y, yerr: mgaussResid(p,x,y)/yerr

# single Gaussian product
# p = [A,mu1,mu2,sigma1,sigma2]
def mgaussisp(x,p,muoffset=[]):
    if len(muoffset) > 0:
        p1 = muoffset[0]
        p2 = muoffset[1]
    else: p1,p2 = 0.,0.
    pi=[p[0],gm_mu([p[1]+p1,p[2]+p1,p[3],p[4]]),gm_sigma([p[3],p[4]])]
    return mgaussi(x,pi)

# super Gaussian normalization
supnorm = lambda sigma,epsilon: (2**(-(3.+epsilon)/(2.+epsilon)))/(sigma*sp.special.gamma(1+(1./(2+epsilon))))

# super Gaussian
# p = [A,mu,sigma,epsilon]
def supgauss(x,p):
    return p[0]*supnorm(p[2],p[3])*sp.exp(-0.5*(abs(x-p[1])/p[2])**(2+p[3]))

# length of beam spot depends also on z1^2+z2^2 and angle and sigmasx
# see Massi equation
def sigma_z(angle,sx1,sx2,zz):
    return 1/np.sqrt(np.sin(angle)**2*(1/sx1**2+1/sx2**2)+((4*np.cos(angle)**2)/(zz)))

# p = [A,mu,sigma]
# p = [A,mu,zz]
def mgaussz(p,angle,w,s,z):
    h = p[0] * mgaussi(z,[w[0]*w[2],p[1],sigma_z(angle,s[0],s[2],p[2])])
    h += p[0] * mgaussi(z,[w[0]*w[3],p[1],sigma_z(angle,s[0],s[3],p[2])])
    h += p[0] * mgaussi(z,[w[1]*w[2],p[1],sigma_z(angle,s[1],s[2],p[2])])
    h += p[0] * mgaussi(z,[w[1]*w[3],p[1],sigma_z(angle,s[1],s[3],p[2])])
    return h

def mgausszResid(p,a,w,s,z,y):
    return mgaussz(p,a,w,s,z)-y
def mgausszResidErr(p,a,w,s,z,y,yerr):
    return mgausszResid(p,a,w,s,z,y)/yerr


# sum of 4 Gaussians
# pdd = [A,w1,w2,w3,mu1,mu2,s1,s2,s3,s4]
mgaussdd = lambda x,p: mgaussi(x,[p[0]*p[1],p[4],p[6]])+mgaussi(x,[p[0]*p[2],p[4],p[7]])+mgaussi(x,[p[0]*p[3],p[5],p[8]])+mgaussi(x,[p[0]*(1-(p[1]+p[2]+p[3])),p[5],p[9]]) 

# parameters for double Gaussian
# rho = w1/w2; A1 = A*w1
# w1 = rho/(1+rho)
# w2 = 1/(1+rho)
# A, w1, mu, sigma1, sigma2
def mgaussd(x,p,muoffset=[]):
    #if len(muoffset) > 0:
    #   p[2] += muoffset[0]
    return mgaussi(x,[p[0]*p[1],p[2],p[3]],muoffset)+mgaussi(x,[p[0]*(1-p[1]),p[2],p[4]],muoffset) # A, w1, mu, sigma1, sigma2
   
mgaussdResid = lambda p,x, y: (mgaussd(x,p)-y)

# A,w,mu,s1,s2
def mgaussdResidErr(p,x, y, yerr):
    #res = mgaussdResid(p,x,y)/yerr

    # don't let the weight p[0] go too small or too large
    # in this case set sigma1 = sigma2
    #if abs((p[1]-0.5)/0.5) > 0.95:
    #    p[4] = p[3]
    if p[3] < 0: p[3] = abs(p[3])
    if p[4] < 0: p[4] = abs(p[4])

    # put
    # sigma1 < sigma2
    p3 = p[3]
    if p[4] < p[3]:
        p[3] = p[4]
        p[4] = p3

    c2,c3,c4 = 0,0,0

    if p[4]/p[3] > 3:
        c2 = ((p[4]/p[3])-3)**2
        
    elif p[3]/p[4] > 3:
        c2 = ((p[3]/p[4])-3)**2
        #print c2

    c4 = abs(((p[3]/p[4])-1))

    # weight close to 1
    #if abs(abs(1-p[1])-0.5) > 0.49:
        #print 'w1',p[1],abs(abs(1-p[1])-0.5),max((p[4]/p[3]),(p[3]/p[4]))
        #p[4] = p[3]
        #c3 = abs(abs(p[4]) - abs(p[3]))/(abs(p[4])+abs(p[3]))/2.

    #c3 = abs(abs(p[4]) - abs(p[3]))/(abs(p[4])+abs(p[3]))/2.
    
    c3 = (max((p[4]/p[3]),(p[3]/p[4]))/2.) - p[1]

    # ratio of sigma close to 1
    #if max((p[4]/p[3]),(p[3]/p[4])) < 1.01:
        #print 'p4/p3',abs(abs(1-p[1])-0.5),max((p[4]/p[3]),(p[3]/p[4]))
        #p[1] = 1

    # sum of weights = 1
    c1 = abs(p[1])+abs(1-p[1]) - 1 # =0
    err = 0.000001


    #print c1,c2,c3,c4
    res = mgaussdResid(p,x,y)/yerr
    #return np.concatenate((res,[c1/err,c2/0.1]))
    return res# + c1/err# + c2/0.1 + c3 + c4


# A,w1,mu,sigma1,sigma2
#mgaussdi = lambda x,p: mgaussi(x,[p[0]*p[1],p[2],p[3]])+mgaussi(x,[p[0]*(1-p[1]),p[2],p[4]])
def mgaussdi(x,p,muoffset=None):
    if muoffset is not None:
        p2 = muoffset[0]
    else:
        p2 = 0.

    p[3] = abs(p[3])
    p[4] = abs(p[4])

    return mgaussi(x,[p[0]*p[1],p[2]+p2,p[3]])+mgaussi(x,[p[0]*(1-p[1]),p[2]+p2,p[4]])

mgaussdiResid = lambda p,x, y: (mgaussdi(x,p)-y)
#mgaussdiResidErr = lambda p,x, y, yerr: mgaussdiResid(p,x,y)/yerr
mgaussdiResidErr = lambda p,x, y, yerr: mgaussdResidErr(p,x,y,yerr)

# product of 2 single Gaussians
# A, mu1,mu2 sigma1, sigma2
mgaussp = lambda x,p: mgaussi(x,[p[0],gm_mu([p[1],p[2],p[3],p[4]]),gm_sigma([p[3],p[4]])])*gnormp([p[1],p[2],p[3],p[4]])

# p = [A,sigma]
mgauss0 = lambda p, x: p[0]*gnorm(p[1])*sp.exp(-(x)**2/(2.0*p[1]**2))
mgauss0Resid = lambda p, x, y: (mgauss0(p,x)-y)
mgauss0ResidErr = lambda p, x, y, yerr: mgauss0Resid(p,x,y)/yerr

# convolved gaussian s = sqrt(s_res^2 + s_beam^2)
# sigma = rms of one resolution
#mgaussconv = lambda p, x,sigma: p[0]*gnorm(np.sqrt(sigma**2+p[2]**2))*sp.exp(-(x-p[1])**2/(2.0*(sigma**2+p[2]**2)))
#mgaussconv = lambda p, x,sigma: mgaussi(x,[p[0],p[1],np.sqrt(sigma**2+p[2]**2)])
def mgaussconv(p,x,sigma,muoffset=[]):
    if len(muoffset) > 0:
        o1 = muoffset[0]
    else:
        o1 = 0.
    if len(p) == 4:
        return mgaussi(x,[p[0],p[1]+o1,np.sqrt((sigma*p[3])**2+p[2]**2)])
    else:
        return mgaussi(x,[p[0],p[1]+o1,np.sqrt(sigma**2+p[2]**2)])

mgaussconvResid = lambda p, x, y,sigmares,muoffset=[]: (mgaussconv(p,x,sigmares,muoffset)-y)
mgaussconvResidErr = lambda p, x, y, yerr,sigmares,muoffset=[]: mgaussconvResid(p,x,y,sigmares,muoffset)/yerr

mgaussconvd = lambda p, x,sigma: mgaussd(x,[p[0],p[1],p[2],np.sqrt(sigma**2+p[3]**2),np.sqrt(sigma**2+p[4]**2)])
mgaussconvdResid = lambda p, x, y,sigmares: (mgaussconvd(p,x,sigmares)-y)
mgaussconvdResidErr = lambda p, x, y, yerr,sigmares: mgaussconvdResid(p,x,y,sigmares)/yerr

rho = lambda s1,s2: s1/s2
rhocorr = lambda rho: (2*rho)/(rho**2 + 1)

dmucorr = lambda dmu,s1,s2: sp.exp(-0.5 * ( (dmu**2)/(s1**2 + s2**2) )  )

trueSigmaZ = lambda szlumi,theta,sxlumi: np.sqrt(np.cos(theta)**2 * 2 * (szlumi**(-2) - (np.tan(theta)**2 * sxlumi**(-2)) )**(-1) )
trueSigmaX = lambda sxlumi,theta: np.sqrt( 2 * np.cos(theta)**2 * sxlumi**2 )

# sigmaz,sigmax are the true width (not lumi spot)
corrtheta = lambda sigmaz,sigmax,theta: (1 + ((sigmaz/sigmax) * np.tan(theta))**2)**(-0.5)

######################
#
#   O V E R L A P integral
#
# slz = Z width of beam spot
# slx = X width of beam spot
# a = half crossing angle
# ss = sigma_z1^2 + sigma_z2^2
def ss_(a,slz,slx):
    return (4 * slx**2 * slz**2 * np.cos(a)**2) / (slx**2 - (slz**2 * np.tan(a)**2))

def ssxy_(ax,ay,slz,slx,sly):
    return 1/(((1/slz**2) - (np.tan(ax)**2 / slx**2) - (np.tan(ay)**2 / sly**2))/(4*np.cos(np.sqrt(ax**2+ay**2))**2))

# overlap integral without angle and head-on
def O0_(sx1,sx2,sy1,sy2):
    return 1/(2 * np.pi * np.sqrt((sx1**2 + sx2**2) * (sy1**2 + sy2**2)))

# correction for crossing angle
def S_(ax,ay,ss,sx1,sx2,sy1,sy2):
    #print ax,ay,ss,sx1,sx2,sy1,sy2
    return (1 + (np.tan(ax)**2 * (ss/(sx1**2 + sx2**2))) + (np.tan(ay)**2 * (ss/(sy1**2 + sy2**2))))**(-1/2.)
def Sx_(ax,ss,sx1,sx2):
    return (1 + (np.tan(ax)**2 * (ss/(sx1**2 + sx2**2))))**(-1/2.)

# correction for displaced beams (not head-on)
def T_(delx,dely,sx1,sx2,sy1,sy2):
    return np.exp(-(delx**2/(2 * (sx1**2+sx2**2)))-dely**2/(2 * (sy1**2+sy2**2)))

# combined term correction for angle and displacement (~1)
def U_(ax,ay,ss,S,delx,dely,sx1,sx2,sy1,sy2):
    return np.exp(S**2 * (ss/2) * ((delx * np.tan(ax))/(sx1**2+sx2**2)+(dely * np.tan(ay))/(sy1**2+sy2**2))**2)

# overlap integral for single Gaussian
#def O_(ax,ay,delx,dely,sx1,sx2,sy1,sy2,slz=0,sz1=0,sz2=0):
def O_(ax,ay,mux1,mux2,muy1,muy2,sx1,sx2,sy1,sy2,slz):

    delx = mux1-mux2
    dely = muy1-muy2
    if 0 in (sx1,sx2,sy1,sy2): return 0
    O0 = O0_(sx1,sx2,sy1,sy2)
    # ss = sqrt(sz1^2 + sz2^2)
    #if sz1 != 0 and sz2 != 0:
        #ss = np.sqrt(sz1**2 + sz2**2)
    #    ss = sz1**2 + sz2**2
    if slz>1000:
        ss = slz
    else:
        slx = gm_sigma([sx1,sx2])
        sly = gm_sigma([sy1,sy2])
        #ss = ss_(ax,slz,slx)
        ss = ssxy_(ax,ay,slz,slx,sly)
    S = S_(ax,ay,ss,sx1,sx2,sy1,sy2)
    T = T_(delx,dely,sx1,sx2,sy1,sy2)
    U = U_(ax,ay,ss,S,delx,dely,sx1,sx2,sy1,sy2)

    return O0*S*T*U

# overlap integral with double Gaussian beams (assuming fully factorizable)
# mu is position at center of beam spot (!= 0)
def Od_(ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,slz):
    # overlap components
    # b1 b2
    # xy xy
    O11_11 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax3,sigmay1,sigmay3,slz)
    O11_12 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax3,sigmay1,sigmay4,slz)
    O12_11 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax3,sigmay2,sigmay3,slz)
    O12_12 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax3,sigmay2,sigmay4,slz)

    O11_21 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax4,sigmay1,sigmay3,slz)
    O11_22 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax4,sigmay1,sigmay4,slz)
    O12_21 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax4,sigmay2,sigmay3,slz)
    O12_22 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax4,sigmay2,sigmay4,slz)

    O21_11 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax3,sigmay1,sigmay3,slz)
    O21_12 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax3,sigmay1,sigmay4,slz)
    O22_11 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax3,sigmay2,sigmay3,slz)
    O22_12 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax3,sigmay2,sigmay4,slz)

    O21_21 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax4,sigmay1,sigmay3,slz)
    O21_22 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax4,sigmay1,sigmay4,slz)
    O22_21 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax4,sigmay2,sigmay3,slz)
    O22_22 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax4,sigmay2,sigmay4,slz)

    # weight components
    wx11 = wx1
    wx12 = 1-wx1
    wx21 = wx2
    wx22 = 1-wx2

    wy11 = wy1
    wy12 = 1-wy1
    wy21 = wy2
    wy22 = 1-wy2

    w11_11 = wx11 * wy11 * wx21 * wy21
    w11_12 = wx11 * wy11 * wx21 * wy22
    w12_11 = wx11 * wy12 * wx21 * wy21
    w12_12 = wx11 * wy12 * wx21 * wy22

    w11_21 = wx11 * wy11 * wx22 * wy21
    w11_22 = wx11 * wy11 * wx22 * wy22
    w12_21 = wx11 * wy12 * wx22 * wy21
    w12_22 = wx11 * wy12 * wx22 * wy22

    w21_11 = wx12 * wy11 * wx21 * wy21
    w21_12 = wx12 * wy11 * wx21 * wy22
    w22_11 = wx12 * wy12 * wx21 * wy21
    w22_12 = wx12 * wy12 * wx21 * wy22

    w21_21 = wx12 * wy11 * wx22 * wy21
    w21_22 = wx12 * wy11 * wx22 * wy22
    w22_21 = wx12 * wy12 * wx22 * wy21
    w22_22 = wx12 * wy12 * wx22 * wy22

    O = 0
    O += w11_11*O11_11
    O += w11_12*O11_12
    O += w12_11*O12_11
    O += w12_12*O12_12

    O += w11_21*O11_21
    O += w11_22*O11_22
    O += w12_21*O12_21
    O += w12_22*O12_22

    O += w21_11*O21_11
    O += w21_12*O21_12
    O += w22_11*O22_11
    O += w22_12*O22_12

    O += w21_21*O21_21
    O += w21_22*O21_22
    O += w22_21*O22_21
    O += w22_22*O22_22

    return O

# overlap integral with double Gaussian beams (using factorizable parameter)
# mu is position at center of beam spot (!= 0)
def Odf_(f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,slz,zzwide=1.):
    # overlap components
    # b1 b2
    # xy xy
    O11_11 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax3,sigmay1,sigmay3,slz)
    O11_12 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax3,sigmay1,sigmay4,slz)
    O12_11 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax3,sigmay2,sigmay3,slz)
    O12_12 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax3,sigmay2,sigmay4,slz)

    O11_21 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax4,sigmay1,sigmay3,slz)
    O11_22 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax4,sigmay1,sigmay4,slz)
    O12_21 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax4,sigmay2,sigmay3,slz)
    O12_22 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax4,sigmay2,sigmay4,slz)

    O21_11 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax3,sigmay1,sigmay3,slz)
    O21_12 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax3,sigmay1,sigmay4,slz)
    O22_11 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax3,sigmay2,sigmay3,slz)
    O22_12 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax3,sigmay2,sigmay4,slz)

    O21_21 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax4,sigmay1,sigmay3,slz)
    O21_22 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax4,sigmay1,sigmay4,slz)
    O22_21 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax4,sigmay2,sigmay3,slz)
    O22_22 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax4,sigmay2,sigmay4,slz*zzwide)

    # weight components
    # weights for fully/non factorizable beams (single beam 1)
    fnn1 = f1 * wx1*wy1         + (1-f1) * (wx1+wy1)/2. # (narrowX,narrowY)
    fnw1 = f1 * wx1*(1-wy1)     + 0
    fwn1 = f1 * wy1*(1-wx1)     + 0
    fww1 = f1 * (1-wx1)*(1-wy1) + (1-f1) * (1-(wx1+wy1)/2.)

    # weights for fully/non factorizable beams (single beam 2)
    fnn2 = f2 * wx2*wy2         + (1-f2) * (wx2+wy2)/2.
    fnw2 = f2 * wx2*(1-wy2)     + 0
    fwn2 = f2 * wy2*(1-wx2)     + 0
    fww2 = f2 * (1-wx2)*(1-wy2) + (1-f2) * (1-(wx2+wy2)/2.)

    #print fnn1
    #print fnw1
    #print fwn1
    #print fww1
    #print fnn2
    #print fnw2
    #print fwn2
    #print fww2
    #print ''

    #B1 B2
    #xy xy
    w11_11 = fnn1 * fnn2
    w11_12 = fnn1 * fnw2
    w12_11 = fnw1 * fnn2
    w12_12 = fnw1 * fnw2

    w11_21 = fnn1 * fwn2
    w11_22 = fnn1 * fww2
    w12_21 = fnw1 * fwn2
    w12_22 = fnw1 * fww2

    w21_11 = fwn1 * fnn2
    w21_12 = fwn1 * fnw2
    w22_11 = fww1 * fnn2
    w22_12 = fww1 * fnw2

    w21_21 = fwn1 * fwn2
    w21_22 = fwn1 * fww2
    w22_21 = fww1 * fwn2
    w22_22 = fww1 * fww2
    #print 'w11_11',f(w11_11,4)
    #print 'w11_12',f(w11_12,4)
    #print 'w12_11',f(w12_11,4)
    #print 'w12_12',f(w12_12,4)
    #print 'w11_21',f(w11_21,4)
    #print 'w11_22',f(w11_22,4)
    #print 'w12_21',f(w12_21,4)
    #print 'w12_22',f(w12_22,4)
    #print 'w21_11',f(w21_11,4)
    #print 'w21_12',f(w21_12,4)
    #print 'w22_11',f(w22_11,4)
    #print 'w22_12',f(w22_12,4)
    #print 'w21_21',f(w21_21,4)
    #print 'w21_22',f(w21_22,4)
    #print 'w22_21',f(w22_21,4)
    #print 'w22_22',f(w22_22,4)

    O = 0
    O += w11_11*O11_11
    O += w11_12*O11_12
    O += w12_11*O12_11
    O += w12_12*O12_12

    O += w11_21*O11_21
    O += w11_22*O11_22
    O += w12_21*O12_21
    O += w12_22*O12_22

    O += w21_11*O21_11
    O += w21_12*O21_12
    O += w22_11*O22_11
    O += w22_12*O22_12

    O += w21_21*O21_21
    O += w21_22*O21_22
    O += w22_21*O22_21
    O += w22_22*O22_22

    return O
# ,zzwide=1. scale factor to zz for the wide component of the beam only
def Od2D_(f1,f2,wx1,wy1,mux1,muy1,s1x,s2x,s1y,s2y,wx2,wy2,mux2,muy2,s3x,s4x,s3y,s4y,ax1,ay1,ax2,ay2,zz,zzwide=1.):
    ax = (ax1 - ax2)/2.
    ay = (ay1 - ay2)/2.    
    return Odf_(f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,s1x,s2x,s3x,s4x,s1y,s2y,s3y,s4y,zz,zzwide)

# length of beam spot depends also on z1^2+z2^2 and angle and sigmasx
# see Massi equation
def sigma_z2(angle,sx1,sx2,zz):
    return 1/np.sqrt(np.sin(angle)**2*(1/sx1**2+1/sx2**2)+((4*np.cos(angle)**2)/(zz)))

# p = [A,mu,sigma]
# p = [A,mu,zz]
# wxl = [px[0],1-px[0],px[1],1-px[1]] weights for X
# wyl = [py[0],1-py[0],py[1],1-py[1]] .. for Y
def mgaussz2(p,angle,wx1,wx2,wy1,wy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z):
    # weight components
    wx11 = wx1
    wx12 = 1-wx1
    wx21 = wx2
    wx22 = 1-wx2

    wy11 = wy1
    wy12 = 1-wy1
    wy21 = wy2
    wy22 = 1-wy2
    # mt.S_(angx,0,10000,sigmax1,sigmax3,sigmay1,sigmay3)
    # beam 1 sigma1 <-> beam 2 sigma 1
    h =  mgaussi(z,[1,p[1],sigma_z(angle,sigmax1,sigmax3,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay1,sigmay3) * S_(angle,0,p[2],sigmax1,sigmax3,sigmay1,sigmay3) * (wx11 * wy11 * wx21 * wy21)
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax1,sigmax3,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay1,sigmay4) * S_(angle,0,p[2],sigmax1,sigmax3,sigmay1,sigmay4) * (wx11 * wy11 * wx21 * wy22)
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax1,sigmax3,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay2,sigmay3) * S_(angle,0,p[2],sigmax1,sigmax3,sigmay2,sigmay3) * (wx11 * wy12 * wx21 * wy21)
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax1,sigmax3,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay2,sigmay4) * S_(angle,0,p[2],sigmax1,sigmax3,sigmay2,sigmay4) * (wx11 * wy12 * wx21 * wy22)
    # beam 1 sigma1 1 * (<-> beam 2 sigma 2)1/1/
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax1,sigmax4,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay1,sigmay3) * S_(angle,0,p[2],sigmax1,sigmax4,sigmay1,sigmay3) * (wx11 * wy11 * wx22 * wy21)
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax1,sigmax4,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay1,sigmay4) * S_(angle,0,p[2],sigmax1,sigmax4,sigmay1,sigmay4) * (wx11 * wy11 * wx22 * wy22)
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax1,sigmax4,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay2,sigmay3) * S_(angle,0,p[2],sigmax1,sigmax4,sigmay2,sigmay3) * (wx11 * wy12 * wx22 * wy21)
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax1,sigmax4,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay2,sigmay4) * S_(angle,0,p[2],sigmax1,sigmax4,sigmay2,sigmay4) * (wx11 * wy12 * wx22 * wy22)
    # beam 1 sigma2 1 * (<-> beam 2 sigma 1)1/1/
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax2,sigmax3,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay1,sigmay3) * S_(angle,0,p[2],sigmax2,sigmax3,sigmay1,sigmay3) * (wx12 * wy11 * wx21 * wy21)
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax2,sigmax3,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay1,sigmay4) * S_(angle,0,p[2],sigmax2,sigmax3,sigmay1,sigmay4) * (wx12 * wy11 * wx21 * wy22)
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax2,sigmax3,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay2,sigmay3) * S_(angle,0,p[2],sigmax2,sigmax3,sigmay2,sigmay3) * (wx12 * wy12 * wx21 * wy21)
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax2,sigmax3,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay2,sigmay4) * S_(angle,0,p[2],sigmax2,sigmax3,sigmay2,sigmay4) * (wx12 * wy12 * wx21 * wy22)
    # beam 1 sigma2 1 * (<-> beam 2 sigma 2)1/1/
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax2,sigmax4,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay1,sigmay3) * S_(angle,0,p[2],sigmax2,sigmax4,sigmay1,sigmay3) * (wx12 * wy11 * wx22 * wy21)
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax2,sigmax4,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay1,sigmay4) * S_(angle,0,p[2],sigmax2,sigmax4,sigmay1,sigmay4) * (wx12 * wy11 * wx22 * wy22)
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax2,sigmax4,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay2,sigmay3) * S_(angle,0,p[2],sigmax2,sigmax4,sigmay2,sigmay3) * (wx12 * wy12 * wx22 * wy21)
    h += mgaussi(z,[1,p[1],sigma_z(angle,sigmax2,sigmax4,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay2,sigmay4) * S_(angle,0,p[2],sigmax2,sigmax4,sigmay2,sigmay4) * (wx12 * wy12 * wx22 * wy22)

    return p[0] * h

def mgausszResid2(p,a,wx1,wx2,wy1,wy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z,y):
    return mgaussz2(p,a,wx1,wx2,wy1,wy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z)-y
def mgausszResidErr2(p,a,wx1,wx2,wy1,wy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z,y,yerr,internerr=True):
    if internerr:
        yerr_ = np.sqrt(mgaussz2(p,a,wx1,wx2,wy1,wy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z))*errampl
        yerr_ = np.average([yerr_,yerr],axis=0)
    else:
        yerr_ = yerr
    return mgausszResid2(p,a,wx1,wx2,wy1,wy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z,y)/yerr_

####
# X and Y angles
# length of beam spot depends also on z1^2+z2^2 and angle and sigmasx
# see Massi equation (14) page 5
def sigma_zxy(ax,ay,sx1,sx2,sy1,sy2,zz):
    return 1/np.sqrt( np.sin(ax)**2*(4/(sx1**2+sx2**2))+np.sin(ay)**2*(4/(sy1**2+sy2**2))+((4*np.cos(np.sqrt(ax**2+ay**2))**2)/(zz)) )

# previous version (without variable integration)
def sigma_zxy_(ax,ay,sx1,sx2,sy1,sy2,zz):
    return 1/np.sqrt(np.sin(ax)**2*(1/sx1**2+1/sx2**2)+np.sin(ay)**2*(1/sy1**2+1/sy2**2)+((4*np.cos(np.sqrt(ax**2+ay**2))**2)/(zz)))

# p = [A,mu,sigma]
# p = [A,mu,zz]
# wxl = [px[0],1-px[0],px[1],1-px[1]] weights for X
# wyl = [py[0],1-py[0],py[1],1-py[1]] .. for Y
# fa is factorization parameter (1= fully factorizable)
def mgausszxy(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z):
    # weight components
    # weight components
    # weights for fully/non factorizable beams (single beam 1)
    fnn1 = f1 * wx1*wy1         + (1-f1) * (wx1+wy1)/2. # (narrowX,narrowY)
    fnw1 = f1 * wx1*(1-wy1)     + 0
    fwn1 = f1 * wy1*(1-wx1)     + 0
    fww1 = f1 * (1-wx1)*(1-wy1) + (1-f1) * (1-(wx1+wy1)/2.)

    # weights for fully/non factorizable beams (single beam 2)
    fnn2 = f2 * wx2*wy2         + (1-f2) * (wx2+wy2)/2.
    fnw2 = f2 * wx2*(1-wy2)     + 0
    fwn2 = f2 * wy2*(1-wx2)     + 0
    fww2 = f2 * (1-wx2)*(1-wy2) + (1-f2) * (1-(wx2+wy2)/2.)

    #B1 B2
    #xy xy
    w11_11 = fnn1 * fnn2
    w11_12 = fnn1 * fnw2
    w12_11 = fnw1 * fnn2
    w12_12 = fnw1 * fnw2

    w11_21 = fnn1 * fwn2
    w11_22 = fnn1 * fww2
    w12_21 = fnw1 * fwn2
    w12_22 = fnw1 * fww2

    w21_11 = fwn1 * fnn2
    w21_12 = fwn1 * fnw2
    w22_11 = fww1 * fnn2
    w22_12 = fww1 * fnw2

    w21_21 = fwn1 * fwn2
    w21_22 = fwn1 * fww2
    w22_21 = fww1 * fwn2
    w22_22 = fww1 * fww2
    # mt.S_(angx,0,10000,sigmax1,sigmax3,sigmay1,sigmay3)                                                                                                                     #B1 B2
    # beam 1 sigma1 <-> beam 2 sigma 1                                                                                                                                        #xy xy          
    h =  mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay1,sigmay3,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay1,sigmay3) * w11_11
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay1,sigmay4,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay1,sigmay4) * w11_12
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay2,sigmay3,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay2,sigmay3) * w12_11
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay2,sigmay4,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay2,sigmay4) * w12_12
    # beam 1 sigma1 1 * (<-> beam 2 sigma 2)1/1/#
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay1,sigmay3,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay1,sigmay3) * w11_21
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay1,sigmay4,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay1,sigmay4) * w11_22
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay2,sigmay3,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay2,sigmay3) * w12_21
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay2,sigmay4,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay2,sigmay4) * w12_22
    # beam 1 sigma2 1 * (<-> beam 2 sigma 1)1/1/#
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay1,sigmay3,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay1,sigmay3) * w21_11
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay1,sigmay4,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay1,sigmay4) * w21_12
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay2,sigmay3,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay2,sigmay3) * w22_11
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay2,sigmay4,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay2,sigmay4) * w22_12
    # beam 1 sigma2 1 * (<-> beam 2 sigma 2)1/1/#
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay1,sigmay3,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay1,sigmay3) * w21_21
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay1,sigmay4,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay1,sigmay4) * w21_22
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay2,sigmay3,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay2,sigmay3) * w22_21
    h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay2,sigmay4,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay2,sigmay4) * w22_22


    # h =  mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay1,sigmay3,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay1,sigmay3) * w11_11
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay1,sigmay4,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay1,sigmay4) * w11_12
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay2,sigmay3,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay2,sigmay3) * w12_11
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay2,sigmay4,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay2,sigmay4) * w12_12
    # # beam 1 sigma1 1 * (<-> beam 2 sigma 2)1/1/
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay1,sigmay3,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay1,sigmay3) * w11_21
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay1,sigmay4,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay1,sigmay4) * w11_22
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay2,sigmay3,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay2,sigmay3) * w12_21
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay2,sigmay4,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay2,sigmay4) * w12_22
    # # beam 1 sigma2 1 * (<-> beam 2 sigma 1)1/1/
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay1,sigmay3,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay1,sigmay3) * w21_11
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay1,sigmay4,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay1,sigmay4) * w21_12
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay2,sigmay3,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay2,sigmay3) * w22_11
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay2,sigmay4,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay2,sigmay4) * w22_12
    # # beam 1 sigma2 1 * (<-> beam 2 sigma 2)1/1/
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay1,sigmay3,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay1,sigmay3) * w21_21
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay1,sigmay4,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay1,sigmay4) * w21_22
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay2,sigmay3,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay2,sigmay3) * w22_21
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay2,sigmay4,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay2,sigmay4) * w22_22

    return p[0] * h

def mgausszxyResid(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z,y):
    return mgausszxy(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z)-y
def mgausszxyResidErr(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z,y,yerr,internerr=True):
    if internerr:
        yerr_ = np.sqrt(mgausszxy(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z))*errampl
        yerr_ = np.average([yerr_,yerr],axis=0)
    else:
        yerr_ = yerr
    return mgausszxyResid(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z,y)/yerr_


####
# deltaZlum: lumi spot displacement from ZRF due to displacement in deltax
# see Massi equation (21) page 5
# only on x coord and angle
def deltaZlum(delmu,a,s1,s2,zz):
    return delmu * (-np.sin(2*a)/4) * ( (zz-(s1**2+s2**2))/( (np.sin(a)**2*zz) + (np.cos(a)**2*(s1**2+s2**2)) ) )
    #return 1/np.sqrt(np.sin(ax)**2*(1/sx1**2+1/sx2**2)+np.sin(ay)**2*(1/sy1**2+1/sy2**2)+((4*np.cos(np.sqrt(ax**2+ay**2))**2)/(zz)))

# LHCb-PUB-2012-016 Eq. 19
def deltaZlum_(delmu,a,s1,s2,zz):
    t = (np.sin(a)*np.cos(a)) * (((2*s1**2*s2**2)/(s1**2+s2**2)) - (zz/2.))
    b = 2 * (( np.sin(a)**2 * (zz/2.) ) + (np.cos(a)**2 * ((2*s1**2*s2**2)/(s1**2+s2**2)) ) )
    return delmu * (t/b)


# p = [A,Zrf,zz][bkg]
def _mgausszxyrf(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z):
    # p[1] = Zrf = Zlum + delZlum
    # p[2] = z1^2+z2^2
    #
    # weight components
    # weights for fully/non factorizable beams (single beam 1)
    fnn1 = f1 * wx1*wy1         + (1-f1) * (wx1+wy1)/2. # (narrowX,narrowY)
    fnw1 = f1 * wx1*(1-wy1)     + 0
    fwn1 = f1 * wy1*(1-wx1)     + 0
    fww1 = f1 * (1-wx1)*(1-wy1) + (1-f1) * (1-(wx1+wy1)/2.)

    # weights for fully/non factorizable beams (single beam 2)
    fnn2 = f2 * wx2*wy2         + (1-f2) * (wx2+wy2)/2.
    fnw2 = f2 * wx2*(1-wy2)     + 0
    fwn2 = f2 * wy2*(1-wx2)     + 0
    fww2 = f2 * (1-wx2)*(1-wy2) + (1-f2) * (1-(wx2+wy2)/2.)

    #B1 B2
    #xy xy
    w11_11 = fnn1 * fnn2
    w11_12 = fnn1 * fnw2
    w12_11 = fnw1 * fnn2
    w12_12 = fnw1 * fnw2

    w11_21 = fnn1 * fwn2
    w11_22 = fnn1 * fww2
    w12_21 = fnw1 * fwn2
    w12_22 = fnw1 * fww2

    w21_11 = fwn1 * fnn2
    w21_12 = fwn1 * fnw2
    w22_11 = fww1 * fnn2
    w22_12 = fww1 * fnw2

    w21_21 = fwn1 * fwn2
    w21_22 = fwn1 * fww2
    w22_21 = fww1 * fwn2
    w22_22 = fww1 * fww2
    # mt.S_(angx,0,10000,sigmax1,sigmax3,sigmay1,sigmay3)                                                                                                                     #B1 B2
    # beam 1 sigma1 <-> beam 2 sigma 1  

    #ax = (ax1 - ax2)/2.
    delx = mux1-mux2
    delxrf = 2 * ax * p[1]
    dmx = delx + delxrf
    dely = muy1-muy2
    delyrf = 2* ay * p[1]
    dmy = dely + delyrf
                                                                                                                               #xy xy          
    h =  mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay1,sigmay3,p[2])]) * O0_(sigmax1,sigmax3,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay1,sigmay3) * w11_11
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay1,sigmay4,p[2])]) * O0_(sigmax1,sigmax3,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay1,sigmay4) * w11_12
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay2,sigmay3,p[2])]) * O0_(sigmax1,sigmax3,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay2,sigmay3) * w12_11
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay2,sigmay4,p[2])]) * O0_(sigmax1,sigmax3,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay2,sigmay4) * w12_12
    # beam 1 sigma1 1 * (<-> beam 2 sigma 2)1/1/#
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay1,sigmay3,p[2])]) * O0_(sigmax1,sigmax4,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay1,sigmay3) * w11_21
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay1,sigmay4,p[2])]) * O0_(sigmax1,sigmax4,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay1,sigmay4) * w11_22
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay2,sigmay3,p[2])]) * O0_(sigmax1,sigmax4,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay2,sigmay3) * w12_21
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay2,sigmay4,p[2])]) * O0_(sigmax1,sigmax4,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay2,sigmay4) * w12_22
    # beam 1 sigma2 1 * (<-> beam 2 sigma 1)1/1/#
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay1,sigmay3,p[2])]) * O0_(sigmax2,sigmax3,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay1,sigmay3) * w21_11
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay1,sigmay4,p[2])]) * O0_(sigmax2,sigmax3,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay1,sigmay4) * w21_12
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay2,sigmay3,p[2])]) * O0_(sigmax2,sigmax3,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay2,sigmay3) * w22_11
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay2,sigmay4,p[2])]) * O0_(sigmax2,sigmax3,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay2,sigmay4) * w22_12
    # beam 1 sigma2 1 * (<-> beam 2 sigma 2)1/1/#
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay1,sigmay3,p[2])]) * O0_(sigmax2,sigmax4,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay1,sigmay3) * w21_21
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay1,sigmay4,p[2])]) * O0_(sigmax2,sigmax4,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay1,sigmay4) * w21_22
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay2,sigmay3,p[2])]) * O0_(sigmax2,sigmax4,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay2,sigmay3) * w22_21
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay2,sigmay4,p[2])]) * O0_(sigmax2,sigmax4,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay2,sigmay4) * w22_22

    if len(p) == 3:
        return p[0] * h
    if len(p) == 4:
        return  (p[0] * h) + abs(p[3]) # background subtraction


# p = [A,Zrf,zz][bkg]
def mgausszxyrf(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z):
    # p[1] = Zrf = Zlum + delZlum
    # p[2] = z1^2+z2^2
    #
    # weight components
    # weights for fully/non factorizable beams (single beam 1)
    fnn1 = f1 * wx1*wy1         + (1-f1) * (wx1+wy1)/2. # (narrowX,narrowY)
    fnw1 = f1 * wx1*(1-wy1)     + 0
    fwn1 = f1 * wy1*(1-wx1)     + 0
    fww1 = f1 * (1-wx1)*(1-wy1) + (1-f1) * (1-(wx1+wy1)/2.)

    # weights for fully/non factorizable beams (single beam 2)
    fnn2 = f2 * wx2*wy2         + (1-f2) * (wx2+wy2)/2.
    fnw2 = f2 * wx2*(1-wy2)     + 0
    fwn2 = f2 * wy2*(1-wx2)     + 0
    fww2 = f2 * (1-wx2)*(1-wy2) + (1-f2) * (1-(wx2+wy2)/2.)

    #B1 B2
    #xy xy
    w11_11 = fnn1 * fnn2
    w11_12 = fnn1 * fnw2
    w12_11 = fnw1 * fnn2
    w12_12 = fnw1 * fnw2

    w11_21 = fnn1 * fwn2
    w11_22 = fnn1 * fww2
    w12_21 = fnw1 * fwn2
    w12_22 = fnw1 * fww2

    w21_11 = fwn1 * fnn2
    w21_12 = fwn1 * fnw2
    w22_11 = fww1 * fnn2
    w22_12 = fww1 * fnw2

    w21_21 = fwn1 * fwn2
    w21_22 = fwn1 * fww2
    w22_21 = fww1 * fwn2
    w22_22 = fww1 * fww2
    # mt.S_(angx,0,10000,sigmax1,sigmax3,sigmay1,sigmay3)                                                                                                                     #B1 B2
    # beam 1 sigma1 <-> beam 2 sigma 1  

    #ax = (ax1 - ax2)/2.
    delx = mux1-mux2
    delxrf = 2 * ax * p[1]
    dmx = delx + delxrf
    dely = muy1-muy2
    delyrf = 2* ay * p[1]
    dmy = dely + delyrf
                                                                                                                               #xy xy          
    h =  mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay1,sigmay3,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay1,sigmay3) * 1/S_(ax,ay,p[2],sigmax1,sigmax3,sigmay1,sigmay3) * w11_11
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay1,sigmay4,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay1,sigmay4) * 1/S_(ax,ay,p[2],sigmax1,sigmax3,sigmay1,sigmay4) * w11_12
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay2,sigmay3,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay2,sigmay3) * 1/S_(ax,ay,p[2],sigmax1,sigmax3,sigmay2,sigmay3) * w12_11
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay2,sigmay4,p[2])]) * 1/O0_(sigmax1,sigmax3,sigmay2,sigmay4) * 1/S_(ax,ay,p[2],sigmax1,sigmax3,sigmay2,sigmay4) * w12_12
    # beam 1 sigma1 1 * (<-> beam 2 sigma 2)1/1/#1/
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay1,sigmay3,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay1,sigmay3) * 1/S_(ax,ay,p[2],sigmax1,sigmax4,sigmay1,sigmay3) * w11_21
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay1,sigmay4,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay1,sigmay4) * 1/S_(ax,ay,p[2],sigmax1,sigmax4,sigmay1,sigmay4) * w11_22
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay2,sigmay3,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay2,sigmay3) * 1/S_(ax,ay,p[2],sigmax1,sigmax4,sigmay2,sigmay3) * w12_21
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax1,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay2,sigmay4,p[2])]) * 1/O0_(sigmax1,sigmax4,sigmay2,sigmay4) * 1/S_(ax,ay,p[2],sigmax1,sigmax4,sigmay2,sigmay4) * w12_22
    # beam 1 sigma2 1 * (<-> beam 2 sigma 1)1/1/#1/
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay1,sigmay3,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay1,sigmay3) * 1/S_(ax,ay,p[2],sigmax2,sigmax3,sigmay1,sigmay3) * w21_11
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay1,sigmay4,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay1,sigmay4) * 1/S_(ax,ay,p[2],sigmax2,sigmax3,sigmay1,sigmay4) * w21_12
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay2,sigmay3,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay2,sigmay3) * 1/S_(ax,ay,p[2],sigmax2,sigmax3,sigmay2,sigmay3) * w22_11
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax3,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay2,sigmay4,p[2])]) * 1/O0_(sigmax2,sigmax3,sigmay2,sigmay4) * 1/S_(ax,ay,p[2],sigmax2,sigmax3,sigmay2,sigmay4) * w22_12
    # beam 1 sigma2 1 * (<-> beam 2 sigma 2)1/1/#1/
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay1,sigmay3,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay1,sigmay3) * 1/S_(ax,ay,p[2],sigmax2,sigmax4,sigmay1,sigmay3) * w21_21
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay1,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay1,sigmay4,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay1,sigmay4) * 1/S_(ax,ay,p[2],sigmax2,sigmax4,sigmay1,sigmay4) * w21_22
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay3,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay2,sigmay3,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay2,sigmay3) * 1/S_(ax,ay,p[2],sigmax2,sigmax4,sigmay2,sigmay3) * w22_21
    h += mgaussi(z,[1,p[1]+deltaZlum(dmx,ax,sigmax2,sigmax4,p[2])+deltaZlum(dmy,ay,sigmay2,sigmay4,p[2]),sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay2,sigmay4,p[2])]) * 1/O0_(sigmax2,sigmax4,sigmay2,sigmay4) * 1/S_(ax,ay,p[2],sigmax2,sigmax4,sigmay2,sigmay4) * w22_22

    if len(p) == 3:
        return p[0] * h
    if len(p) == 4:
        return  (p[0] * h) + abs(p[3]) # single background subtraction
    if len(p) == 5:
        idx1 = np.where(z<0)[0]
        idx2 = np.where(z>=0)[0]
        h *= p[0]
        h[idx1] += abs(p[3])
        h[idx2] += abs(p[4])
        return  h # per beam background subtraction

def mgausszxyrfResid(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z,y):
    return mgausszxyrf(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z)-y
def mgausszxyrfResidErr(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z,y,yerr,internerr=True):
    if internerr:
        yerr_ = np.sqrt(mgausszxyrf(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z))*errampl
        #yerr_ = np.average([yerr_,yerr],axis=0)
        yerr_ = np.average([yerr_,yerr],axis=0,weights=[100,1])

    else:
        yerr_ = yerr
    return mgausszxyrfResid(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z,y)/yerr_


"""
def mgausszxy(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z):
    # weight components
    # weight components
    # weights for fully/non factorizable beams (single beam 1)
    fnn1 = f1 * wx1*wy1         + (1-f1) * (wx1+wy1)/2. # (narrowX,narrowY)
    fnw1 = f1 * wx1*(1-wy1)     + 0
    fwn1 = f1 * wy1*(1-wx1)     + 0
    fww1 = f1 * (1-wx1)*(1-wy1) + (1-f1) * (1-(wx1+wy1)/2.)

    # weights for fully/non factorizable beams (single beam 2)
    fnn2 = f2 * wx2*wy2         + (1-f2) * (wx2+wy2)/2.
    fnw2 = f2 * wx2*(1-wy2)     + 0
    fwn2 = f2 * wy2*(1-wx2)     + 0
    fww2 = f2 * (1-wx2)*(1-wy2) + (1-f2) * (1-(wx2+wy2)/2.)

    #B1 B2
    #xy xy
    w11_11 = fnn1 * fnn2
    w11_12 = fnn1 * fnw2
    w12_11 = fnw1 * fnn2
    w12_12 = fnw1 * fnw2

    w11_21 = fnn1 * fwn2
    w11_22 = fnn1 * fww2
    w12_21 = fnw1 * fwn2
    w12_22 = fnw1 * fww2

    w21_11 = fwn1 * fnn2
    w21_12 = fwn1 * fnw2
    w22_11 = fww1 * fnn2
    w22_12 = fww1 * fnw2

    w21_21 = fwn1 * fwn2
    w21_22 = fwn1 * fww2
    w22_21 = fww1 * fwn2
    w22_22 = fww1 * fww2

    O11_11 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax3,sigmay1,sigmay3,p[2])
    O11_12 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax3,sigmay1,sigmay4,p[2])
    O12_11 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax3,sigmay2,sigmay3,p[2])
    O12_12 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax3,sigmay2,sigmay4,p[2])

    O11_21 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax4,sigmay1,sigmay3,p[2])
    O11_22 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax4,sigmay1,sigmay4,p[2])
    O12_21 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax4,sigmay2,sigmay3,p[2])
    O12_22 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax1,sigmax4,sigmay2,sigmay4,p[2])

    O21_11 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax3,sigmay1,sigmay3,p[2])
    O21_12 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax3,sigmay1,sigmay4,p[2])
    O22_11 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax3,sigmay2,sigmay3,p[2])
    O22_12 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax3,sigmay2,sigmay4,p[2])

    O21_21 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax4,sigmay1,sigmay3,p[2])
    O21_22 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax4,sigmay1,sigmay4,p[2])
    O22_21 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax4,sigmay2,sigmay3,p[2])
    O22_22 = O_(ax,ay,mux1,mux2,muy1,muy2,sigmax2,sigmax4,sigmay2,sigmay4,p[2])

    h = 0
    h += w11_11*O11_11 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay1,sigmay3,p[2])]) 
    h += w11_12*O11_12 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay1,sigmay4,p[2])])
    h += w12_11*O12_11 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay2,sigmay3,p[2])])
    h += w12_12*O12_12 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay2,sigmay4,p[2])])

    h += w11_21*O11_21 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay1,sigmay3,p[2])])
    h += w11_22*O11_22 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay1,sigmay4,p[2])])
    h += w12_21*O12_21 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay2,sigmay3,p[2])])
    h += w12_22*O12_22 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay2,sigmay4,p[2])])

    h += w21_11*O21_11 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay1,sigmay3,p[2])])
    h += w21_12*O21_12 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay1,sigmay4,p[2])])
    h += w22_11*O22_11 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay2,sigmay3,p[2])])
    h += w22_12*O22_12 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay2,sigmay4,p[2])])

    h += w21_21*O21_21 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay1,sigmay3,p[2])])
    h += w21_22*O21_22 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay1,sigmay4,p[2])])
    h += w22_21*O22_21 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay2,sigmay3,p[2])])
    h += w22_22*O22_22 * mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay2,sigmay4,p[2])])

    #print 'w11_11*O11_11',w11_11*O11_11 
    #print 'w11_12*O11_12',w11_12*O11_12
    #print 'w12_11*O12_11',w12_11*O12_11
    #print 'w12_12*O12_12',w12_12*O12_12

    #print 'w11_21*O11_21',w11_21*O11_21
    #print 'w11_22*O11_22',w11_22*O11_22
    #print 'w12_21*O12_21',w12_21*O12_21
    #print 'w12_22*O12_22',w12_22*O12_22

    #print 'w21_11*O21_11',w21_11*O21_11
    #print 'w21_12*O21_12',w21_12*O21_12
    #print 'w22_11*O22_11',w22_11*O22_11
    #print 'w22_12*O22_12',w22_12*O22_12

    #print 'w21_21*O21_21',w21_21*O21_21
    #print 'w21_22*O21_22',w21_22*O21_22
    #print 'w22_21*O22_21',w22_21*O22_21
    #print 'w22_22*O22_22',w22_22*O22_22

    # mt.S_(angx,0,10000,sigmax1,sigmax3,sigmay1,sigmay3)

    # h =  mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay1,sigmay3,p[2])]) * 1/(O0_(sigmax1,sigmax3,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay1,sigmay3)) * w11_11
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay1,sigmay4,p[2])]) * 1/(O0_(sigmax1,sigmax3,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay1,sigmay4)) * w11_12
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay2,sigmay3,p[2])]) * 1/(O0_(sigmax1,sigmax3,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay2,sigmay3)) * w12_11
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax3,sigmay2,sigmay4,p[2])]) * 1/(O0_(sigmax1,sigmax3,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax3,sigmay2,sigmay4)) * w12_12

    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay1,sigmay3,p[2])]) * 1/(O0_(sigmax1,sigmax4,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay1,sigmay3)) * w11_21
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay1,sigmay4,p[2])]) * 1/(O0_(sigmax1,sigmax4,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay1,sigmay4)) * w11_22
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay2,sigmay3,p[2])]) * 1/(O0_(sigmax1,sigmax4,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay2,sigmay3)) * w12_21
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax1,sigmax4,sigmay2,sigmay4,p[2])]) * 1/(O0_(sigmax1,sigmax4,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax1,sigmax4,sigmay2,sigmay4)) * w12_22

    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay1,sigmay3,p[2])]) * 1/(O0_(sigmax2,sigmax3,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay1,sigmay3)) * w21_11
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay1,sigmay4,p[2])]) * 1/(O0_(sigmax2,sigmax3,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay1,sigmay4)) * w21_12
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay2,sigmay3,p[2])]) * 1/(O0_(sigmax2,sigmax3,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay2,sigmay3)) * w22_11
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax3,sigmay2,sigmay4,p[2])]) * 1/(O0_(sigmax2,sigmax3,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax3,sigmay2,sigmay4)) * w22_12

    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay1,sigmay3,p[2])]) * 1/(O0_(sigmax2,sigmax4,sigmay1,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay1,sigmay3)) * w21_21
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay1,sigmay4,p[2])]) * 1/(O0_(sigmax2,sigmax4,sigmay1,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay1,sigmay4)) * w21_22
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay2,sigmay3,p[2])]) * 1/(O0_(sigmax2,sigmax4,sigmay2,sigmay3) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay2,sigmay3)) * w22_21
    # h += mgaussi(z,[1,p[1],sigma_zxy(ax,ay,sigmax2,sigmax4,sigmay2,sigmay4,p[2])]) * 1/(O0_(sigmax2,sigmax4,sigmay2,sigmay4) * S_(ax,ay,p[2],sigmax2,sigmax4,sigmay2,sigmay4)) * w22_22

    return p[0] * h

def mgausszxyResid(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z,y):
    return mgausszxy(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z)-y
def mgausszxyResidErr(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z,y,yerr,internerr=False):
    if internerr:
        yerr_ = np.sqrt(mgausszxy(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z))*errampl
        yerr_ = np.average([yerr_,yerr],axis=0)
    else:
        yerr_ = yerr
    return mgausszxyResid(p,f1,f2,ax,ay,wx1,wx2,wy1,wy2,mux1,mux2,muy1,muy2,sigmax1,sigmax2,sigmax3,sigmax4,sigmay1,sigmay2,sigmay3,sigmay4,z,y)/yerr_
"""
######################
def poisson(n,mu):
    return (np.exp(-mu)*mu**n)/factorial(n)

"""
# using Gaussian sigma = sqrt(sn^2+s^2)
# weight list relative number of entries in a Gaussian e/etot
# sigmas list of velo resolutions (1 per Gaussian)
def mgaussmultconv(p, x, weight, sigmas):
    p[2] = abs(p[2]) # force positive sigma
    result = 0
    for i in range(0,len(sigmas)):
        result += (weight[i] * mgaussconv(p, x, sigma=sigmas[i]))
    return result
def mgaussmultconvResidErr(p, x, y,yerr,weight,sigmas):
    return mgaussmultconvResid(p, x, y,weight,sigmas)/yerr
def mgaussmultconvResid(p, x, y,weight,sigmas):
    return mgaussmultconv(p,x,weight,sigmas)-y
"""

# Using gaussians
# sa list relative number of entries in a Gaussian e/etot
# sn list of velo resolutions (1 per Gaussian)

# using Gaussian sigma = sqrt(sn^2+s^2)
# weight list relative number of entries in a Gaussian e/etot
# sigmas list of velo resolutions (1 per Gaussian)
def mgaussmultconv(p, x, resolution,muoffset=[],fn=None):

    p[2] = abs(p[2]) # force positive sigma
    weight,sigmas = resolution
    result = 0
    for i in range(0,len(sigmas)):
        result += (weight[i] * mgaussconv(p,x,sigmas[i],muoffset))
    return result

def mgaussmultconvResid(p, x, y,resolution,muoffset=[],fn=None):
    return mgaussmultconv(p,x,resolution,muoffset,fn)-y
# [str(round(pi, i)) for i in range(1, 6)]
# ['3.1', '3.14', '3.142', '3.1416', '3.14159']
def mgaussmultconvResidErr(p, x, y,yerr,resolution,muoffset=[],fn=None,nint=0,internerr=False):
    p[2] = abs(p[2]) # force positive sigma

    if internerr:
        yerr_ = np.sqrt(mgaussmultconv(p,x,resolution,muoffset))*errampl
        yerr_ = np.average([yerr_,yerr],axis=0)
    else:
        yerr_ = yerr

    # nint = number of bins inside x to integrate y
    if nint != 0 and hasattr(x, '__iter__'):
        dx = x[1]-x[0]
        dxi = dx/nint
        return np.average([mgaussmultconvResid(p, x-(dx/2.)+(i*dxi)+(dxi/2.), y,resolution,muoffset)/yerr_ for i in range(nint)],axis=0)
    else:
        return mgaussmultconvResid(p, x, y,resolution,muoffset)/yerr_

# single Gaussian product
# p = [mu1,mu2,sigma1,sigma2,A]
def mgaussmultconvspResidErr(p, x, y,yerr,resolution,muoffset=[],fn=None,nint=0,internerr=True):
    if len(muoffset) == 2:
        o1 = muoffset[0]
        o2 = muoffset[1]
    else:
        o1,o2 = 0,0

    # this is the new mu. -> don't propagate muofset
    pi=[p[0],gm_mu([p[1]+o1,p[2]+o2,p[3],p[4]]),gm_sigma([p[3],p[4]])]

    if internerr:
        yerr_ = np.sqrt(mgaussmultconv(pi,x,resolution))*errampl
        yerr_ = np.average([yerr_,yerr],axis=0)
    else:
        yerr_ = yerr

    if nint != 0 and hasattr(x, '__iter__'):
        dx = x[1]-x[0]
        dxi = dx/nint
        return np.average([mgaussmultconvResid(pi, x-(dx/2.)+(i*dxi)+(dxi/2.), y,resolution)/yerr_ for i in range(nint)],axis=0)
    else:
        return mgaussmultconvResid(pi, x, y,resolution)/yerr_

def mgaussmultconvsp(p, x, resolution,muoffset=[]):
    if len(muoffset) == 2:
        o1 = muoffset[0]
        o2 = muoffset[1]
    else:
        o1,o2 = 0,0

    pi=[p[0],gm_mu([p[1]+o1,p[2]+o2,p[3],p[4]]),gm_sigma([p[3],p[4]])]
    weight,sigmas = resolution
    result = 0
    for i in range(0,len(sigmas)):
        result += (weight[i] * mgaussconv(pi, x, sigmas[i]))
    return result

# double Gaussian beam
# convolve each (main) Gaussian and summ at the end
#      0  1  2  3  4   5   6  7  8  9 # 10 11 # resolution scale
# p = [A,w1,w2,w3,mu1,mu2,s1,s2,s3,s4]#rs1,rs2
def mgaussmultconvdd(p, x, resolution):
    weight,sigmas = resolution
    r1,r2,r3,r4 = 0,0,0,0

    if len(p) == 12:
        for i in range(0,len(sigmas)):
            r1 += (mgaussconv([weight[i]*p[0]*p[1],p[4],p[6],p[10]], x, sigma=sigmas[i]))
            r2 += (mgaussconv([weight[i]*p[0]*p[2],p[4],p[7],p[10]], x, sigma=sigmas[i]))
            r3 += (mgaussconv([weight[i]*p[0]*p[3],p[5],p[8],p[11]], x, sigma=sigmas[i]))
            r4 += (mgaussconv([weight[i]*p[0]*(1-(p[1]+p[2]+p[3])),p[5],p[9],p[11]], x, sigma=sigmas[i]))
    else: # no res scale
        for i in range(0,len(sigmas)):
            r1 += (mgaussconv([weight[i]*p[0]*p[1],p[4],p[6]], x, sigma=sigmas[i]))
            r2 += (mgaussconv([weight[i]*p[0]*p[2],p[4],p[7]], x, sigma=sigmas[i]))
            r3 += (mgaussconv([weight[i]*p[0]*p[3],p[5],p[8]], x, sigma=sigmas[i]))
            r4 += (mgaussconv([weight[i]*p[0]*(1-(p[1]+p[2]+p[3])),p[5],p[9]], x, sigma=sigmas[i]))

    return r1+r2+r3+r4

def mgaussmultconvddResidErr(p, x, y,yerr,resolution):
    return mgaussmultconvddResid(p, x, y,resolution)/yerr
def mgaussmultconvddResid(p, x, y,resolution):
    return mgaussmultconvdd(p,x,resolution)-y

# 2 double Gaussian product convolution
def mgaussmultconvddp(p, x, resolution):
    weight,sigmas = resolution
    result = 0
    for i in range(0,len(sigmas)):
        result += weight[i]*mgaussddp(x,[p[0],p[1],p[2],p[3],p[4],np.sqrt(sigmas[i]**2+p[5]**2),np.sqrt(sigmas[i]**2+p[6]**2),np.sqrt(sigmas[i]**2+p[7]**2),np.sqrt(sigmas[i]**2+p[8]**2)])

    return result
def mgaussmultconvddpResidErr(p, x, y,yerr,resolution):
    res = mgaussmultconvddpResid(p, x, y,resolution)/yerr
    c = (p[1]*p[2])+(p[1]*(1-p[2]))+((1-p[1])*p[2])+((1-p[1])*(1-p[2])) - 1 #= 0
    c1 = abs(p[1])+abs(1-p[1]) - 1 # =0
    c2 = abs(p[2])+abs(1-p[2]) - 1 # =0
    return np.concatenate((res,[c,c1,c2]))

def mgaussmultconvddpResid(p, x, y,resolution):
    return mgaussmultconvddp2(p,x,resolution)-y
    
# double Gaussian beam
# convolve each (main) Gaussian and summ at the end # p[5]= res factor (optional)
#      0 1  2    3      4     5
# p = [A,w1,mu,sigma1,sigma2]#rs (optional resolution scale)
def mgaussmultconvd(p, x, resolution,muoffset=[]):

    weight,sigmas = resolution
    r1,r2 = 0,0

    if len(p) == 6:
        #global itercount1D
        #try: itercount1D += 1
        #except: itercount1D = 0
        #if itercount1D%2000 == 0:
        #    print 'mgaussmultconvd',p

        for i in range(0,len(sigmas)):
            r1 += (mgaussconv([weight[i]*p[0]*p[1],p[2],p[3],p[5]], x, sigmas[i],muoffset))
            r2 += (mgaussconv([weight[i]*p[0]*(1-p[1]),p[2],p[4],p[5]], x, sigmas[i],muoffset))
    else:
        for i in range(0,len(sigmas)):
            r1 += (mgaussconv([weight[i]*p[0]*p[1],p[2],p[3]], x, sigmas[i],muoffset))
            r2 += (mgaussconv([weight[i]*p[0]*(1-p[1]),p[2],p[4]], x, sigmas[i],muoffset))
    return r1+r2

    # p [A,w1,mu,s1,s2]
def mgaussmultconvdResidErr(p, x, y,yerr,resolution,muoffset=[],fn=None,nint=0,internerr=True):
 


    #res = mgaussmultconvdResid(p, x, y,resolution)/yerr
    if internerr:
        yerr_ = np.sqrt(abs(mgaussmultconvd(p,x,resolution,muoffset)))*errampl
        yerr_ = np.average([yerr_,yerr],axis=0)
    else:
        yerr_ = yerr

    if nint != 0 and hasattr(x, '__iter__'):
        dx = x[1]-x[0]
        dxi = dx/nint
        res = np.average([mgaussmultconvdResid(p, x-(dx/2.)+(i*dxi)+(dxi/2.), y,resolution,muoffset)/yerr_ for i in range(nint)],axis=0)
    else:
        res = mgaussmultconvdResid(p, x, y,resolution,muoffset)/yerr_

    #return res + c1/err# + c2/0.1 + c3/0.1 + c4/0.001
    #return np.concatenate((res,c1/err))
    return res


def mgaussmultconvdResid(p, x, y,resolution,muoffset=[]):
    return mgaussmultconvd(p,x,resolution,muoffset)-y


def mgaussmultsi(x,p, resolution):
    weight,sigmas = resolution
    result = 0
    for i in range(0,len(sigmas)):
        result += (weight[i] * mgauss([p[0],p[1],sigmas[i]], x))
    return result

def mgaussmults(p, x, resolution):
    weight,sigmas = resolution
    result = 0
    for i in range(0,len(sigmas)):
        result += (weight[i] * mgauss([p[0],p[1],sigmas[i]], x))
    return result
def mgaussmult(p, x):
    result = 0
    for i in range(0,len(p)/3):
        result += (mgauss(p[i*3:i*3+3], x))
    return result
def mgaussmultResid(p, x, y):
    return mgaussmult(p,x)-y
def mgaussmultResidErr(p, x, y,yerr):
    return mgaussmultResid(p,x,y)/yerr

# prod of 2x2 Gaussians (without convolution)
#        [0,  1, 2, 3, 4,  5,  6,  7,  8]
# pddp = [A1,w1,w2,mu1,mu2,sa1,sa2,sb1,sb2]
def mgaussddp3(x,p,muoffset=[]):
    return p[0]*sum(mgaussddp3list(x,p,muoffset))
"""    
    if len(muoffset) == 2:
        o1 = muoffset[0]
        o2 = muoffset[1]
    else:
        o1,o2 = 0,0
   
    y  = mgaussi(x,[(1-p[1])*(1-p[2]),gm_mu([p[3]+o1,p[4]+o2,p[6],p[8]]),gm_sigma([p[6],p[8]])])*gnormp([p[3]+o1,p[4]+o2,p[6],p[8]])
    y += mgaussi(x,[(1-p[1])*p[2],gm_mu([p[3]+o1,p[4]+o2,p[6],p[7]]),gm_sigma([p[6],p[7]])])*gnormp([p[3]+o1,p[4]+o2,p[6],p[7]])
    y += mgaussi(x,[p[1]*(1-p[2]),gm_mu([p[3]+o1,p[4]+o2,p[5],p[8]]),gm_sigma([p[5],p[8]])])*gnormp([p[3]+o1,p[4]+o2,p[5],p[8]])
    y += mgaussi(x,[p[1]*p[2],gm_mu([p[3]+o1,p[4]+o2,p[5],p[7]]),gm_sigma([p[5],p[7]])])*gnormp([p[3]+o1,p[4]+o2,p[5],p[7]])

    return p[0]*y
"""
#        [0,  1, 2, 3, 4,  5,  6,  7,  8]
# pddp = [A1,w1,w2,mu1,mu2,sa1,sa2,sb1,sb2]
# return list of 4 gauss for one axis, beam1 x beam2
def mgaussddp3list(x,p,muoffset=[]):
    if len(muoffset) == 2:
        o1 = muoffset[0]
        o2 = muoffset[1]
    else:
        o1,o2 = 0,0
   
    ww = mgaussi(x,[(1-p[1])*(1-p[2]),gm_mu([p[3]+o1,p[4]+o2,p[6],p[8]]),gm_sigma([p[6],p[8]])])*gnormp([p[3]+o1,p[4]+o2,p[6],p[8]])
    wn = mgaussi(x,[(1-p[1])*p[2],gm_mu([p[3]+o1,p[4]+o2,p[6],p[7]]),gm_sigma([p[6],p[7]])])*gnormp([p[3]+o1,p[4]+o2,p[6],p[7]])
    nw = mgaussi(x,[p[1]*(1-p[2]),gm_mu([p[3]+o1,p[4]+o2,p[5],p[8]]),gm_sigma([p[5],p[8]])])*gnormp([p[3]+o1,p[4]+o2,p[5],p[8]])
    nn = mgaussi(x,[p[1]*p[2],gm_mu([p[3]+o1,p[4]+o2,p[5],p[7]]),gm_sigma([p[5],p[7]])])*gnormp([p[3]+o1,p[4]+o2,p[5],p[7]])

    return nn,nw,wn,ww

mgaussddp3Resid = lambda p,x, y,muoffset=[]: (mgaussddp3(x,p,muoffset)-y)
def mgaussddp3ResidErr(p,x, y, yerr,muoffset=[]):
    return mgaussddp3Resid(p,x,y,muoffset)/yerr

#########
#
# 2 double Gaussian product convolution
#        [0,  1, 2, 3,  4,  5,  6,  7,  8]
# pddp = [A1,w1,w2,mu1,mu2,sa1,sa2,sb1,sb2]
def mgaussmultconvddp3(p, x, resolution,muoffset=[]):
    return p[0]*sum(mgaussmultconvddp3list(p, x, resolution,muoffset))

def mgaussmultconvddp3list(p, x, resolution,muoffset=[]):
    if len(muoffset) > 0:
        o1 = muoffset[0]
        o2 = muoffset[1]
    else:
        o1,o2 = 0,0

    weight,sigmas = resolution
    nn,nw,wn,ww = 0,0,0,0

    # this is correct for 1D fit
    # for 2D fit, all weights must be set to 0.5 before (as to not double use the weights, see line 1304)
    for i in range(0,len(sigmas)):
        ww += weight[i]*mgaussi(x,[(1-p[1])*(1-p[2]),gm_mu([p[3]+o1,p[4]+o2,p[6],p[8]]),np.sqrt(sigmas[i]**2+gm_sigma([p[6],p[8]])**2)])*gnormp([p[3]+o1,p[4]+o2,p[6],p[8]])
        wn += weight[i]*mgaussi(x,[(1-p[1])*p[2]    ,gm_mu([p[3]+o1,p[4]+o2,p[6],p[7]]),np.sqrt(sigmas[i]**2+gm_sigma([p[6],p[7]])**2)])*gnormp([p[3]+o1,p[4]+o2,p[6],p[7]])
        nw += weight[i]*mgaussi(x,[p[1]*(1-p[2])    ,gm_mu([p[3]+o1,p[4]+o2,p[5],p[8]]),np.sqrt(sigmas[i]**2+gm_sigma([p[5],p[8]])**2)])*gnormp([p[3]+o1,p[4]+o2,p[5],p[8]])
        nn += weight[i]*mgaussi(x,[p[1]*p[2]        ,gm_mu([p[3]+o1,p[4]+o2,p[5],p[7]]),np.sqrt(sigmas[i]**2+gm_sigma([p[5],p[7]])**2)])*gnormp([p[3]+o1,p[4]+o2,p[5],p[7]])

    #print p
    return nn,nw,wn,ww

def test():
    print test

# pddp = [A1,w1,w2,mu1,mu2,sa1,sa2,sb1,sb2]
def mgaussmultconvddp3ResidErr(p, x, y,yerr,resolution,muoffset=[],fn=None,nint=0,internerr=True):        
    #global itercount1D
    #try: itercount1D += 1
    #except: itercount1D = 0
    #if itercount1D%2000 == 0:
    #    print 'mgaussmultconvddp3ResidErr',p
    
    if internerr:
        yerr_ = np.sqrt(mgaussmultconvddp3(p,x,resolution,muoffset))*errampl
        yerr_ = np.average([yerr_,yerr],axis=0)
    else:
        yerr_ = yerr
    

    if nint != 0 and hasattr(x, '__iter__'):
        dx = x[1]-x[0]
        dxi = dx/nint
        res = np.average([mgaussmultconvddp3Resid(p, x-(dx/2.)+(i*dxi)+(dxi/2.), y,resolution,muoffset)/yerr_ for i in range(nint)],axis=0)
    else:
        res = mgaussmultconvddp3Resid(p, x, y,resolution,muoffset)/yerr_

    return res


def mgaussmultconvddp3Resid(p, x, y,resolution,muoffset=[]):
    return mgaussmultconvddp3(p,x,resolution,muoffset)-y


# prod of 2x2 Gaussians (without convolution) in 2D
# angles are optional. If present, muoffset is z position of histogram
# in this case use angles and z to calculate the mu offset -> angles are fit parameters
# new muoffset is axi * zp
#      0  1  2   3   4    5    6   7   8   9   10  11  12   13  14  15  16  17  18  19 20  21  22
# p = [A,f1,f2,wx1,wy1,mux1,muy1,s1x,s2x,s1y,s2y,wx2,wy2,mux2,muy2,s3x,s4x,s3y,s4y,ax1,ay1,ax2,ay2
def mgaussddp2D(p_,x,y,muoffset=[],fn=None):
    if fn is None:
        p=p_
    else:
        p=[0]*(len(p_)+2)
        p[0] = p_[0]
        p[1] = fn
        p[2] = fn
        p[3:]=p_[1:]


    if len(muoffset) == 4:
        ox1 = muoffset[0]
        oy1 = muoffset[1]
        ox2 = muoffset[2]
        oy2 = muoffset[3]
    else:
        ox1,oy1,ox2,oy2 = 0,0,0,0

    if len(p) == 23:
        zp = muoffset[-1]
        ox1 = zp * p[19]
        oy1 = zp * p[20]
        ox2 = zp * p[21]
        oy2 = zp * p[22]


    # weights for fully/non factorizable beams (single beam 1)
    fnn1 = p[1] * p[3]*p[4]         + (1-p[1]) * (p[3]+p[4])/2. # (narrowX,narrowY)
    fnw1 = p[1] * p[3]*(1-p[4])     + 0
    fwn1 = p[1] * p[4]*(1-p[3])     + 0
    fww1 = p[1] * (1-p[3])*(1-p[4]) + (1-p[1]) * (1-(p[3]+p[4])/2.)

    # weights for fully/non factorizable beams (single beam 2)
    fnn2 = p[2] * p[11]*p[12]         + (1-p[2]) * (p[11]+p[12])/2.
    fnw2 = p[2] * p[11]*(1-p[12])     + 0
    fwn2 = p[2] * p[12]*(1-p[11])     + 0
    fww2 = p[2] * (1-p[11])*(1-p[12]) + (1-p[2]) * (1-(p[11]+p[12])/2.)

    #px = [1,p[2],p[10],p[4]+ox1,p[12]+ox2,p[6],p[7],p[14],p[15]] # pddp X [A1,w1,w2,mu1,mu2,sa1,sa2,sb1,sb2]
    #py = [1,p[3],p[11],p[5]+oy1,p[13]+oxy,p[8],p[9],p[16],p[17]] # pddp Y [A1,w1,w2,mu1,mu2,sa1,sa2,sb1,sb2]
    # mgaussddp3list uses the weights already (for 1D fit), but here we treat the weights before
    # -> get mgaussddp3list with equal weights
    px = [1,0.5,0.5,p[5]+ox1,p[13]+ox2,p[7],p[8],p[15],p[16]] # pddp X [A1,w1,w2,mu1,mu2,sa1,sa2,sb1,sb2]
    py = [1,0.5,0.5,p[6]+oy1,p[14]+oy2,p[9],p[10],p[17],p[18]] # pddp Y [A1,w1,w2,mu1,mu2,sa1,sa2,sb1,sb2]

    # components for product of beam 1 X beam 2 per axis (nnx = narrowX1 x narrowX2)
    #B1B2
    nnx,nwx,wnx,wwx = mgaussddp3list(x,px) # array of len x
    nny,nwy,wny,wwy = mgaussddp3list(y,py)

    #     B1     B2    X     Y
    r=0#  xy     xy    12    12
    r += fnn1 * fnn2 * nnx * nny
    r += fnn1 * fnw2 * nnx * nwy
    r += fnn1 * fwn2 * nwx * nny
    r += fnn1 * fww2 * nwx * nwy

    r += fnw1 * fnn2 * nnx * wny
    r += fnw1 * fnw2 * nnx * wwy
    r += fnw1 * fwn2 * nwx * wny
    r += fnw1 * fww2 * nwx * wwy

    r += fwn1 * fnn2 * wnx * nny
    r += fwn1 * fnw2 * wnx * nwy
    r += fwn1 * fwn2 * wwx * nny
    r += fwn1 * fww2 * wwx * nwy

    r += fww1 * fnn2 * wnx * wny
    r += fww1 * fnw2 * wnx * wwy
    r += fww1 * fwn2 * wwx * wny
    r += fww1 * fww2 * wwx * wwy

    #soaw = fnn1*fnn2+fnn1*fnw2+fnn1*fwn2+fnn1*fww2+ fnw1*fnn2+fnw1*fnw2+fnw1*fwn2+fnw1*fww2+ fwn1*fnn2+fwn1*fnw2+fwn1*fwn2+fwn1*fww2+ fww1*fnn2+fww1*fnw2+fww1*fwn2+fww1*fww2

    return p[0]*r


def mgaussddp2DResid(p,x,y,data,muoffset=[]):
    return mgaussddp2D(p,x,y,muoffset) - data
#       0 1  2   3   4    5    6   7   8   9   10  11  12   13  14  15  16  17  18
# p = [A,f1,f2,wx1,wy1,mux1,muy1,s1x,s2x,s1y,s2y,wx2,wy2,mux2,muy2,s3x,s4x,s3y,s4y]
def mgaussddp2DResidErr(p,x,y,data,err,muoffset=[]):

    c1 = constrainFactor([p[1],p[3],p[4]]) # p = [fn,wx,wy]
    c2 = constrainFactor([p[2],p[11],p[12]]) # p = [fn,wx,wy]

    ra1 = constrainSigmaRatios([p[3],p[7],p[8]]) # p = [w,sigmax1,sigmax2]
    ra2 = constrainSigmaRatios([p[4],p[9],p[10]])
    ra3 = constrainSigmaRatios([p[11],p[15],p[16]])
    ra4 = constrainSigmaRatios([p[12],p[17],p[18]])

    res = mgaussddp2DResid(p,x,y,data,muoffset)/err

    return np.concatenate((res,c1,c2,ra1,ra2,ra3,ra4))

# for fitting amplitude only
def mgaussddp2Dampl(a,p,x,y,muoffset=[]):
    p[0]=a[0]
    return mgaussddp2D(p,x,y,muoffset)
def mgaussddp2DResidampl(a,p,x,y,data,muoffset=[]):
    return mgaussddp2Dampl(a,p,x,y,muoffset) - data
#      0  1  2   3   4    5    6   7   8   9   10  11  12   13  14  15  16  17 18
# p = [A,f1,f2,wx1,wy1,mux1,muy1,s1x,s2x,s1y,s2y,wx2,wy2,mux2,muy2,s3x,s4x,s3y,s4y]
def mgaussddp2DResidErrampl(a,p,x,y,data,err,muoffset=[],constr=False):
    res = mgaussddp2DResidampl(a,p,x,y,data,muoffset)/err
    if not constr:
        return res
    else:
        c1 = constrainFactor([p[1],p[3],p[4]]) # p = [fn,wx,wy]
        c2 = constrainFactor([p[2],p[11],p[12]]) # p = [fn,wx,wy]

        ra1 = constrainSigmaRatios([p[3],p[7],p[8]]) # p = [w,sigmax1,sigmax2]
        ra2 = constrainSigmaRatios([p[4],p[9],p[10]])
        ra3 = constrainSigmaRatios([p[11],p[15],p[16]])
        ra4 = constrainSigmaRatios([p[12],p[17],p[18]])

        return np.concatenate((res,c1,c2,ra1,ra2,ra3,ra4))

# prod of 2x2 Gaussians (with convolution) in 2D
#       0 1  2   3   4    5    6   7   8   9   10  11  12   13  14  15  16  17  18
# p = [A,f1,f2,wx1,wy1,mux1,muy1,s1x,s2x,s1y,s2y,wx2,wy2,mux2,muy2,s3x,s4x,s3y,s4y]
def mgaussddp2Dconv(p_,x,y,resx,resy,muoffset=[],fn=None):
    if fn is None:
        p=p_
    else:
        p=[0]*(len(p_)+2)
        p[0] = p_[0]
        p[1] = fn
        p[2] = fn
        p[3:]=p_[1:]

    #print 'mgaussddp2Dconv',len(p),p,muoffset
    if len(muoffset) >= 4 and len(p) == 19:
        ox1 = muoffset[0] # B1
        oy1 = muoffset[1]
        ox2 = muoffset[2] # B2
        oy2 = muoffset[3]
    else:
        ox1,oy1,ox2,oy2 = 0,0,0,0
        #print muoffset,ox1,oy1,ox2,oy2

    if len(p) == 23: # fit also angles
        zp = muoffset[-1]
        ox1 = zp * p[19]
        oy1 = zp * p[20]
        ox2 = zp * p[21]
        oy2 = zp * p[22]
        #print muoffset,ox1,oy1,ox2,oy2


    # weights for fully/non factorizable beams (single beam 1)
    fnn1 = p[1] * p[3]*p[4]         + (1-p[1]) * (p[3]+p[4])/2. # (narrowX,narrowY)
    fnw1 = p[1] * p[3]*(1-p[4])     + 0
    fwn1 = p[1] * p[4]*(1-p[3])     + 0
    fww1 = p[1] * (1-p[3])*(1-p[4]) + (1-p[1]) * (1-(p[3]+p[4])/2.)

    # weights for fully/non factorizable beams (single beam 2)
    fnn2 = p[2] * p[11]*p[12]         + (1-p[2]) * (p[11]+p[12])/2.
    fnw2 = p[2] * p[11]*(1-p[12])     + 0
    fwn2 = p[2] * p[12]*(1-p[11])     + 0
    fww2 = p[2] * (1-p[11])*(1-p[12]) + (1-p[2]) * (1-(p[11]+p[12])/2.)

    #px = [1,p[2],p[10],p[4]+ox1,p[12]+ox2,p[6],p[7],p[14],p[15]] # pddp X [A1,w1,w2,mu1,mu2,sa1,sa2,sb1,sb2]
    #py = [1,p[3],p[11],p[5]+oy1,p[13]+oxy,p[8],p[9],p[16],p[17]] # pddp Y [A1,w1,w2,mu1,mu2,sa1,sa2,sb1,sb2]
    # mgaussddp3list uses the weights already (for 1D fit), but here we treat the weights before
    # -> get mgaussddp3list with equal weights
    px = [1,0.5,0.5,p[5]+ox1,p[13]+ox2,p[7],p[8],p[15],p[16]] # pddp X [A1,w1,w2,mu1,mu2,sa1,sa2,sb1,sb2]
    py = [1,0.5,0.5,p[6]+oy1,p[14]+oy2,p[9],p[10],p[17],p[18]] # pddp Y [A1,w1,w2,mu1,mu2,sa1,sa2,sb1,sb2]


    nnx,nwx,wnx,wwx = mgaussmultconvddp3list(px,x,resx) # array of len x
    nny,nwy,wny,wwy = mgaussmultconvddp3list(py,y,resy)


    r=0#  xy     xy    12    12
    r += fnn1 * fnn2 * nnx * nny
    r += fnn1 * fnw2 * nnx * nwy
    r += fnn1 * fwn2 * nwx * nny
    r += fnn1 * fww2 * nwx * nwy

    r += fnw1 * fnn2 * nnx * wny
    r += fnw1 * fnw2 * nnx * wwy
    r += fnw1 * fwn2 * nwx * wny
    r += fnw1 * fww2 * nwx * wwy

    r += fwn1 * fnn2 * wnx * nny
    r += fwn1 * fnw2 * wnx * nwy
    r += fwn1 * fwn2 * wwx * nny
    r += fwn1 * fww2 * wwx * nwy

    r += fww1 * fnn2 * wnx * wny
    r += fww1 * fnw2 * wnx * wwy
    r += fww1 * fwn2 * wwx * wny
    r += fww1 * fww2 * wwx * wwy

    #soaw = fnn1*fnn2+fnn1*fnw2+fnn1*fwn2+fnn1*fww2+ fnw1*fnn2+fnw1*fnw2+fnw1*fwn2+fnw1*fww2+ fwn1*fnn2+fwn1*fnw2+fwn1*fwn2+fwn1*fww2+ fww1*fnn2+fww1*fnw2+fww1*fwn2+fww1*fww2

    return p[0]*r

def mgaussddp2DconvResid(p,x,y,data,resx,resy,muoffset=[],fn=None):
    return mgaussddp2Dconv(p,x,y,resx,resy,muoffset,fn)-data
def mgaussddp2DconvResidErr(p,x,y,data,err,resx,resy,muoffset=[],fn=None,nint=0):
    #if nint != 0 and hasattr(x, '__iter__'):
    #    dx = x[1]-x[0]
    #    dxi = dx/nint
    #    res = np.average([mgaussmultconvddp3Resid(p, x-(dx/2.)+(i*dxi)+(dxi/2.), y,resolution,muoffset)/yerr_ for i in range(nint)],axis=0)
    #else:
    #    res = mgaussddp2DconvResid(p,x,y,data,resx,resy,muoffset)/err

    return mgaussddp2DconvResid(p,x,y,data,resx,resy,muoffset,fn)/err

# p = [fa,A,wx1,wy1,mux1,muy1,s1x,s2x,s1y,s2y,wx2,wy2,mux2,muy2,s3x,s4x,s3y,s4y]
def mgaussddp2DconvResidErrAmpl(a,p,x,y,data,err,resx,resy,muoffset=[]):
    p[0]=a[0]
    return mgaussddp2DconvResidErr(p,x,y,data,err,resx,resy,muoffset)

# single beam 2D triple Gaussian (for MC study)
# assume beam non factorizable
#       0 1 2  3   4   5   6   7   8   9  10  11  12  13  14 15
# p = [A,f1,wx,wx2,wy,wy2,mux,muy,s1x,s2x,s3x,s1y,s2y,s3x,ax,ay
def mgauss3g2D(p,x,y,muoffset=[]):
    if len(muoffset) >= 2:
        ox = muoffset[0]
        oy = muoffset[1]
    else:
        ox,oy = 0,0


    nn = mgaussi(x,[1.,p[6]+ox,p[8]])*mgaussi(y,[1.,p[7]+oy,p[11]])
    mm = mgaussi(x,[1.,p[6]+ox,p[9]])*mgaussi(y,[1.,p[7]+oy,p[12]])
    ww = mgaussi(x,[1.,p[6]+ox,p[10]])*mgaussi(y,[1.,p[7]+oy,p[13]])

    # weights for fully/non factorizable beams (single beam 1)
    fnn = (p[2]+p[4])/2. # (narrowX,narrowY)
    fmm = (p[3]+p[5])/2.
    fww = 1-(fnn+fmm)

    res = fnn*nn + fmm*mm + fww*ww

    return p[0]*res


# single beam 2D super Gaussian
#      0  1   2   3   4     5       6
# p = [A,mux,muy,s1x,s1y,epsilonx,epsilony]
def mgaussSg2D(p,x,y,muoffset=[]):
    if len(muoffset) >= 2:
        ox = muoffset[0]
        oy = muoffset[1]
    else:
        ox,oy = 0,0

    # super Gaussian
    # p = [A,mu,sigma,epsilon]
    res = supgauss(x,[1.,p[1]+ox,p[3],p[5]])*supgauss(y,[1.,p[2]+oy,p[4],p[6]])

    return p[0]*res


# single beam 2D
#       0 1 2  3   4   5   6   7   8   9  10 11
# p = [A,f1,wx,wy,mux,muy,s1x,s2x,s1y,s2y,ax,ay
def mgaussd2D(p_,x,y,muoffset=[],fn=None):
    if fn is None:
        p=p_
    else:
        p=[0]*(len(p_)+1)
        p[0] = p_[0]
        p[1] = fn
        p[2:]=p_[1:]

    if len(muoffset) >= 2:
        ox = muoffset[0]
        oy = muoffset[1]
    else:
        ox,oy = 0,0

    if len(p) == 12:
        zp = muoffset[-1]
        ox = zp * p[10]
        oy = zp * p[11]


    nn = mgaussi(x,[1.,p[4]+ox,p[6]])*mgaussi(y,[1.,p[5]+oy,p[8]])
    nw = mgaussi(x,[1.,p[4]+ox,p[6]])*mgaussi(y,[1.,p[5]+oy,p[9]])
    wn = mgaussi(x,[1.,p[4]+ox,p[7]])*mgaussi(y,[1.,p[5]+oy,p[8]])
    ww = mgaussi(x,[1.,p[4]+ox,p[7]])*mgaussi(y,[1.,p[5]+oy,p[9]])

    # weights for fully/non factorizable beams (single beam 1)
    fnn = p[1] * p[2]*p[3]         + (1-p[1]) * (p[2]+p[3])/2. # (narrowX,narrowY)
    fnw = p[1] * p[2]*(1-p[3])     + 0
    fwn = p[1] * p[3]*(1-p[2])     + 0
    fww = p[1] * (1-p[2])*(1-p[3]) + (1-p[1]) * (1-(p[2]+p[3])/2.)

    res = fnn*nn + fnw*nw + fwn*wn + fww*ww

    return p[0]*res

def mgaussd2Dresid(p,x,y,data,muoffset=[]):
    return mgaussd2D(p,x,y,muoffset) - data
def mgaussd2DresidErr(p,x,y,data,err,muoffset=[],constr=False):
    res = mgaussd2Dresid(p,x,y,data,muoffset)/err
    if not constr:
        return res
    else:
        ctr = constrainFactor([p[1],p[2],p[3]])
    return np.concatenate((res,ctr))

# single beam 2D with resolution convolution
#      0 1  2  3   4   5   6   7   8   9 #10  11  12  13
# p = [A,f1,wx,wy,mux,muy,s1x,s2x,s1y,s2y]ax, ay, rsx,rsy
def mgaussconvd2D(p_,x,y,resx,resy,muoffset=[],fn=None):
    if fn is None:
        p=p_
    else:
        p=[0]*(len(p_)+1)
        p[0] = p_[0]
        p[1] = fn
        p[2:]=p_[1:]


    if len(muoffset) >= 2 or len(p) == 10:
        ox = muoffset[0]
        oy = muoffset[1]
        #print 'mgaussconvd2D offsets:',ox,oy
    else:
        ox,oy = 0,0
        #print muoffset,ox,oy

    if len(p) >= 12:
        zp = muoffset[-1]
        ox = zp * p[10]
        oy = zp * p[11]
        #print muoffset,ox,oy

    weightx,sigmasx = resx
    weighty,sigmasy = resy

    r = 0
    nn,nw,wn,ww = 0,0,0,0
    sow = 0 # sum of weights

    if len(p) == 14:
        for ix in range(0,len(sigmasx)):
            for iy in range(0,len(sigmasy)):
                nn += mgaussi(x,[weightx[ix],p[4]+ox,np.sqrt((sigmasx[ix]*p[12])**2+p[6]**2)])*mgaussi(y,[weighty[iy],p[5]+oy,np.sqrt((sigmasy[iy]*p[13])**2+p[8]**2)])
                nw += mgaussi(x,[weightx[ix],p[4]+ox,np.sqrt((sigmasx[ix]*p[12])**2+p[6]**2)])*mgaussi(y,[weighty[iy],p[5]+oy,np.sqrt((sigmasy[iy]*p[13])**2+p[9]**2)])
                wn += mgaussi(x,[weightx[ix],p[4]+ox,np.sqrt((sigmasx[ix]*p[12])**2+p[7]**2)])*mgaussi(y,[weighty[iy],p[5]+oy,np.sqrt((sigmasy[iy]*p[13])**2+p[8]**2)])
                ww += mgaussi(x,[weightx[ix],p[4]+ox,np.sqrt((sigmasx[ix]*p[12])**2+p[7]**2)])*mgaussi(y,[weighty[iy],p[5]+oy,np.sqrt((sigmasy[iy]*p[13])**2+p[9]**2)])
    else: # default no res scale
        for ix in range(0,len(sigmasx)):
            for iy in range(0,len(sigmasy)):
                nn += mgaussi(x,[weightx[ix],p[4]+ox,np.sqrt(sigmasx[ix]**2+p[6]**2)])*mgaussi(y,[weighty[iy],p[5]+oy,np.sqrt(sigmasy[iy]**2+p[8]**2)])
                nw += mgaussi(x,[weightx[ix],p[4]+ox,np.sqrt(sigmasx[ix]**2+p[6]**2)])*mgaussi(y,[weighty[iy],p[5]+oy,np.sqrt(sigmasy[iy]**2+p[9]**2)])
                wn += mgaussi(x,[weightx[ix],p[4]+ox,np.sqrt(sigmasx[ix]**2+p[7]**2)])*mgaussi(y,[weighty[iy],p[5]+oy,np.sqrt(sigmasy[iy]**2+p[8]**2)])
                ww += mgaussi(x,[weightx[ix],p[4]+ox,np.sqrt(sigmasx[ix]**2+p[7]**2)])*mgaussi(y,[weighty[iy],p[5]+oy,np.sqrt(sigmasy[iy]**2+p[9]**2)])

            #sow += weightx[ix]*weighty[iy]
    

    # weights for fully/non factorizable beams (single beam 1)
    fnn = p[1] * p[2]*p[3]         + (1-p[1]) * (p[2]+p[3])/2. # (narrowX,narrowY)
    fnw = p[1] * p[2]*(1-p[3])     + 0
    fwn = p[1] * p[3]*(1-p[2])     + 0
    fww = p[1] * (1-p[2])*(1-p[3]) + (1-p[1]) * (1-(p[2]+p[3])/2.)

    #print p
    #print fnn
    #print fnw
    #print fwn
    #print fww

    res = fnn*nn + fnw*nw + fwn*wn + fww*ww

    return p[0]*res

def mgaussconvd2Dresid(p,x,y,data,resx,resy,muoffset=[],fn=None):
    return mgaussconvd2D(p,x,y,resx,resy,muoffset,fn)-data
# single beam 2D with resolution convolution
#      0 1  2  3   4   5   6   7   8   9
# p = [A,f1,wx,wy,mux,muy,s1x,s2x,s1y,s2y]
def mgaussconvd2DresidErr(p,x,y,data,err,resx,resy,muoffset=[],fn=None,constr=False):
    res = mgaussconvd2Dresid(p,x,y,data,resx,resy,muoffset,fn)/err
    if not constr or len(p) == 9:
        return res
    else:
        ctrf = constrainFactor([p[1],p[2],p[3]])
        ctrx = constrainSigmaRatios([p[2],p[6],p[7]]) # p = [w,sigmax1,sigmax2]
        ctry = constrainSigmaRatios([p[3],p[8],p[9]]) # p = [w,sigmax1,sigmax2]
        global itercount2D
        try: itercount2D += 1
        except: itercount2D = 0
        if itercount2D%400 == 0:
            print constr,'i',itercount2D,'chi^2/dof:',sum(res**2)/(len(x)-len(p))#,'p',p[1:]
            print 'ctrf',ctrf,'ctrx',ctrx,'ctry',ctry
    return np.concatenate((res,ctrf,ctrx,ctry))

# force non factorizability (fa=0)
# p = [A,f1=0,wx,wy,mux,muy,s1x,s2x,s1y,s2y]
def mgaussconvd2D0residErr(p,x,y,data,err,resx,resy,muoffset=[],constr=False):
    p0=[0]*(len(p)+1)
    p0[0] = p[0]
    p0[1] = [0]
    p0[2:]=p[1:]
    return mgaussconvd2DresidErr(p0,x,y,data,err,resx,resy,muoffset,constr)



# prod of 2x2 Gaussians (without convolution)
#        [0,  1, 2, 3, 4,  5,  6,  7,  8     9    10]
# pddp = [A1,w1,w2,mu1,mu2,sa1,sa2,sb1,sb2,ang1,ang2]
def beamSpotPosition(z,p):
    #print z,p
    mu1 = p[9]*z + p[3]
    mu2 = p[10]*z + p[4]

    # beam spot position is weighted average of all mus
    w1 = ((1-p[1])*(1-p[2])) * gnormp([mu1,mu2,p[6],p[8]])
    w2 = ((1-p[1])*p[2])*gnormp([mu1,mu2,p[6],p[7]])
    w3 = (p[1]*(1-p[2]))*gnormp([mu1,mu2,p[5],p[8]])
    w4 = (p[1]*p[2])*gnormp([mu1,mu2,p[5],p[7]])

    m1 = gm_mu([mu1,mu2,p[6],p[8]])
    m2 = gm_mu([mu1,mu2,p[6],p[7]])
    m3 = gm_mu([mu1,mu2,p[5],p[8]])
    m4 = gm_mu([mu1,mu2,p[5],p[7]])

    #print z,[m1,m2,m3,m4],[w1,w2,w3,w4]

    mu = np.average([m1,m2,m3,m4],weights=[w1,w2,w3,w4])
    return mu

# prod of 2x2 Gaussians (without convolution)
#        [0,  1, 2, 3, 4,  5,  6,  7,  8     9    10]
# pddp = [A1,w1,w2,mu1,mu2,sa1,sa2,sb1,sb2,ang1,ang2]
def beamSpotWidth(z,p):
    #print z,p
    mu1 = p[9]*z + p[3]
    mu2 = p[10]*z + p[4]

    # beam spot position is weighted average of all mus
    w1 = ((1-p[1])*(1-p[2])) * gnormp([mu1,mu2,p[6],p[8]])
    w2 = ((1-p[1])*p[2])*gnormp([mu1,mu2,p[6],p[7]])
    w3 = (p[1]*(1-p[2]))*gnormp([mu1,mu2,p[5],p[8]])
    w4 = (p[1]*p[2])*gnormp([mu1,mu2,p[5],p[7]])

    m1 = gm_mu([mu1,mu2,p[6],p[8]])
    m2 = gm_mu([mu1,mu2,p[6],p[7]])
    m3 = gm_mu([mu1,mu2,p[5],p[8]])
    m4 = gm_mu([mu1,mu2,p[5],p[7]])

    mu = np.average([m1,m2,m3,m4],weights=[w1,w2,w3,w4])

    sig1 = np.sqrt(gm_sigma([p[6],p[8]])**2 + (m1-mu)**2)
    sig2 = np.sqrt(gm_sigma([p[6],p[7]])**2 + (m2-mu)**2)
    sig3 = np.sqrt(gm_sigma([p[5],p[8]])**2 + (m3-mu)**2)
    sig4 = np.sqrt(gm_sigma([p[5],p[7]])**2 + (m4-mu)**2)
    
    sig = np.average([sig1,sig2,sig3,sig4],weights=[w1,w2,w3,w4])

    return sig



def corrgnormp(p):
    c = (1-p[1])*(1-p[2])*gnormp([p[3],p[4],p[6],p[8]])
    c += (1-p[1])*p[2]*gnormp([p[3],p[4],p[6],p[7]])
    c += p[1]*(1-p[2])*gnormp([p[3],p[4],p[5],p[8]])
    c += p[1]*p[2]*gnormp([p[3],p[4],p[5],p[7]])

    return c



def multiFuncErr(p,functionl,pl,xl,resolutionl,ol=[]):
    # pl is indeces of p

    yval = []
    p = np.array(p)
    for fi,i in zip(functionl,range(len(functionl))):
        #print p
        #print pl[i]
        if ol==[]:
            yvali = fi(p[pl[i]],xl[i],resolutionl[i])
        else:
            #print f,i,p[pl[i]],xl[i],resolutionl[i],ol[i]
            yvali = fi(p[pl[i]],xl[i],resolutionl[i],ol[i])

        yval.append(np.sqrt(abs(yvali))+1/np.sqrt(sum(yvali)))

    return yval

def multiFuncErr2D(p,functionl,pl,xl,yl,resxl,resyl,ol=[],fn=None):

    yval = []
    p = np.array(p)
    for fi,i in zip(functionl,range(len(functionl))):
        if ol==[]:
            if yl[i] == []: # 1d2d fit
                yvali = fi(p[pl[i]],xl[i],resxl[i])
            else:
                yvali = fi(p[pl[i]],xl[i],yl[i],resxl[i],resyl[i],[],fn)
        else:
            if yl[i] == []:
                yvali = fi(p[pl[i]],xl[i],resxl[i],ol[i])
            else:
                yvali = fi(p[pl[i]],xl[i],yl[i],resxl[i],resyl[i],ol[i],fn)

        yval.append(np.sqrt(abs(yvali))+1/np.sqrt(sum(yvali)))

    return yval

def multiFuncResiduals(p,functionl,pl,xl,yl,yerrl,resolutionl,ol=[],fn=None,nint=0,internerr=True):
    # pl is indeces of p
    res = np.array([])
    p = np.array(p)
    for fi,i in zip(functionl,range(len(functionl))):
        if ol==[]:
            resi = fi(p[pl[i]],xl[i],yl[i],yerrl[i],resolutionl[i],[],fn,nint,internerr)
        else:
            resi = fi(p[pl[i]],xl[i],yl[i],yerrl[i],resolutionl[i],ol[i],fn,nint,internerr)

        #print 'res',res
        #print 'resi',resi
        res = np.concatenate((res,resi))

    return res

def multiFuncResidualsAmplitudes(p,pall,functionl,pl,xl,yl,yerrl,resolutionl,ol=[],fn=None,nint=0,internerr=True):
    pall[-len(p):]=p
    pall=np.array(pall)
    return multiFuncResiduals(pall,functionl,pl,xl,yl,yerrl,resolutionl,ol,fn,nint,internerr)
    """
    res = np.array([])
    for f,i in zip(functionl,range(len(functionl))):
        if ol==[]:
            resi = f(pall[pl[i]],xl[i],yl[i],yerrl[i],resolutionl[i],[],nint,internerr)
        else:
            resi = f(pall[pl[i]],xl[i],yl[i],yerrl[i],resolutionl[i],ol[i],nint,internerr)
        res = np.concatenate((res,resi))

    return res
    """

def multiFuncResiduals2D(p,functionl,pl,xl,yl,datal,errl,resxl,resyl,ol=[],fn=None,nint=0,internerr=True):

    res = np.array([])
    p = np.array(p)
    for fi,i in zip(functionl,range(len(functionl))):
        if ol==[]:
            if yl[i] == []:
                resi = fi(p[pl[i]],xl[i],datal[i],errl[i],resxl[i],[],fn,nint,internerr) # for 1d2d
            else:
                resi = fi(p[pl[i]],xl[i],yl[i],datal[i],errl[i],resxl[i],resyl[i],[],fn,nint)#,nint,internerr)
        else:
            if yl[i] == []:
                resi = fi(p[pl[i]],xl[i],datal[i],errl[i],resxl[i],ol[i],fn,nint,internerr)
            else:
                resi = fi(p[pl[i]],xl[i],yl[i],datal[i],errl[i],resxl[i],resyl[i],ol[i],fn,nint)#,nint,internerr)
        
        res = np.concatenate((res,np.hstack(resi)))

    global itercount2D
    try: itercount2D += 1
    except: itercount2D = 0
    if itercount2D%100 == 0:
        #i = 1
        #print pl[i]
        #print p[pl[i]]
        #print xl[i]
        #print datal[i]
        #print ol[i]
        #print yl[i]
        #print functionl[i](p[pl[i]],xl[i],datal[i],errl[i],resxl[i],ol[i],fn,nint,internerr)
        #print functionl[i]

        print '\niter',itercount2D,'chi^2/dof:',sum(res**2)/(len(np.hstack(xl))-len(p)),p[:10],'fn =',fn
        #print res
        if len(p) > 18:
            #if fn is None:
            #    p_=p
            #else:
            #    p_=[0]*(len(p)+2)
            #    p_[0] = fn
            #    p_[1] = fn
            #    p_[2:]=p[:]

            print 'f1         ',f(p[0],2)
            print 'f2         ',f(p[1],2)
            print 'wx1,s1x,s2x',f(p[2],3),f(p[6],3),f(p[7],3)
            print 'wy1,s1y,s2y',f(p[3],3),f(p[8],3),f(p[9],3)
            print 'wx2,s3x,s4x',f(p[10],3),f(p[14],3),f(p[15],3)
            print 'wy2,s3y,s4y',f(p[11],3),f(p[16],3),f(p[17],3)
            print 'mux1,muy1  ',f(p[4],2),f(p[5],2)
            print 'mux2,muy2  ',f(p[12],2),f(p[13],2)
            if p[18] < 0.5 and len(p)>23: # ax1,ay1,ax2,ay2
                print 'ax1, ay1   ',f(p[18],2),f(p[19],2)
                print 'ax2, ay2   ',f(p[20],2),f(p[21],2)
            if p[22] < 1.5 and len(p)>25:
                print 'rx1, ry1   ',f(p[22],3),f(p[23],3)
                print 'rx2, ry2   ',f(p[24],3),f(p[25],3)


    return res

def multiFuncResiduals2DAmplitudes(p,pall,functionl,pl,xl,yl,datal,errl,resxl,resyl,ol=[],fn=None,nint=0,internerr=True):
    pall[-len(p):]=p
    pall=np.array(pall)
    return multiFuncResiduals2D(pall,functionl,pl,xl,yl,datal,errl,resxl,resyl,ol,fn,nint,internerr)

def multiFuncResidualsScalar(p,functionl,pl,xl,yl,yerrl,resolutionl,ol=[],fn=None):
    return np.sum(multiFuncResiduals(p,functionl,pl,xl,yl,yerrl,resolutionl,ol=[])**2,fn)



def mu_bb(s1,s2,m1,m2):
    return ((m1*s2**2) + (m2*s1**2))/(s1**2 + s2**2)
def sigma_bb(s1,s2):
    return np.sqrt((s1**2 * s2**2)/(s1**2 + s2**2))
## take 6 measurements of beam spot width on CMS(ATLAS), ALICE, LHCb
# and solve the individual beam width
#
# measurements order:
#   IP1       IP2       IP8
#    0         1         2
#    3         4         5
#
# parameters order:
#  p0 p1     p0 p5     p2 p1
#  p4 p3     p2 p3     p4 p5
#
# @param p is parameters list of unknown
# @param y is list of measured values (sigma_bb)
def sigma_b(p):
    fa = np.array(
        [sigma_bb(p[0],p[1]),
        sigma_bb(p[0],p[5]),
        sigma_bb(p[2],p[1]),
        sigma_bb(p[4],p[3]),
        sigma_bb(p[2],p[3]),
        sigma_bb(p[4],p[5])])
    return fa

## residuals of sigma_b
def sigma_bResid(p,y,yerr):
    return (sigma_b(p)-y)/yerr

# y = s1,s2,sbb,mu1,mu2,mubb
# p = s1,s2,mu1,mu2
# works if p,y are numpy arrays: p=np.array(p)
# does not work if p is a list
def constrainResidErr(p, y, yerr,mu=True):
    # p are the parameters to fit 2 sigmas 2 mus
    # y are the measured values 3 sigmas 3 mus
    # f are the computed values
    if not mu: # only sigma are used    
        fa = np.array([p[0],p[1],sigma_bb(p[0],p[1]),\
                      p[2],p[3],y[5]])
    else:
        fa = np.array([p[0],p[1],sigma_bb(p[0],p[1]),\
                      p[2],p[3],mu_bb(p[0],p[1],p[2],p[3])])
    return (fa-y)/yerr

# double Gausian (1 weight)
def constrainWeight(p,*args):
    c1 = abs(p[0])+abs(1-p[0]) - 1 # =0
    err = 0.001
    return [c1/err]

# product of double Gaussian
# p = [w1,w2]
def constrainWeights(p,*args):
    c = (p[0]*p[1])+(p[0]*(1-p[1]))+((1-p[0])*p[1])+((1-p[0])*(1-p[1])) - 1 #= 0
    # force weight 0<w<0
    c1 = abs(p[0])+abs(1-p[0]) - 1 # =0
    c2 = abs(p[1])+abs(1-p[1]) - 1 # =0
    # allow negative weight and also w>1
    c1 = p[0]+1-p[0] - 1 # =0 duh!
    c2 = p[1]+1-p[1] - 1 # =0
    err = 0.0001
    return [c1/err,c2/err,c/err]

# p = [sigma1,sigma2,sigma3,sigma4]
# or
# p = [sigma1,sigma2]
def constrainSigmas(p,*args):
    c = [0.]*len(p)
    for i in range(len(p)):
        if p[i] < 0: c[i] = p[i]
    err = 0.001
    return np.array(c)/err

# p = [w,sigmax1,sigmax2]
def constrainSigmaRatios(p,*args):
    # discourage large 2nd Gaussian if fraction is small
    c2 = 0
    err = 0.001

    r = (max(abs(p[1]),abs(p[2]))/min(abs(p[1]),abs(p[2]))) # ratio of sigmas (>1), =1 for SG
    #fa = abs(abs(1-p[0])-0.5)**4
    fa = abs(abs(1-p[0])-0.5)**3 # small if w close to 0.5, =0.125 at 0 and 1
    if p[0] > 1: fa = 0
    #sc4 = (r-1)*fa*60
    c4 = (r-1)*fa*200
    
    # avoid too large difference between sigmas
    c3 = (((max(abs(p[1]),abs(p[2]))/min(abs(p[1]),abs(p[2])))/3.)**3)
    #c3 = (max(abs(p[1]),abs(p[2]))/min(abs(p[1]),abs(p[2])))*fa*100

    # try using rms instead
    # discourage weight of zero if single Gaussian
    rms = np.sqrt(abs(p[0]*p[1]**2+(1-p[0])*p[2]**2))

    if p[0] > 0.5:
        s = p[1] # s is main sigma (with largest weight)
    else:
        s = p[2]

    rs = abs(1-(rms/s))+0.005 # small if single Gaussian
    #rd = (1/p[0]**2) * (1/rs)
    #rd = ((1-p[0])**2) * (1/rs)
    #rd = ((1-p[0]+0.01)**2) * (1/rs)
    rs2 = 1/(abs((rms/s)-1+0.01)**2) # large is SG
    rd2 = ((1-p[0]+0.01)) # small if p[0]=1

    rs3 = (r-1+0.01)**2

    rd = rs2*((1-p[0]+0.01)**2)

    #r = (max(abs(p[1]),abs(p[2]))/min(abs(p[1]),abs(p[2])))
    #dif = abs(abs(p[1]-p[2])/((p[1]+p[2])/2.))

    #c1=(rd/1000)* abs(rms/s)

    #c1=(rs2*rd2)/100
    c1=(1/(rs3)*(1-p[0]))/100/p[0]



    #f = abs(max(p[0],1-p[0])/abs(min(p[0],1-p[0]))) - 1
    #f *= abs(p[0]-(1-p[0]))
    if p[0] > 1: # negative weight
        #c2 += ((p[0]-1+0.2)**2 )/rs
        #c2 += (((p[0]-1+0.05)**2)*5)/rs
        #c2 += (((p[0]-1+0.05)**2)*40.)/(rs+0.1)
        c2 += (((p[0]-1+0.05)**2)*5.)/(rs+0.1)

        #fb = p[1]/p[2]-1+0.01 # add penalty if sigmas are equal (and weight>1)
        #c2 /= fb**2
        # 1st sigma must be larger than second
        if abs(p[1]) < abs(p[2]):
            #c2 += (1-(abs(p[2])/abs(p[1])))*10000
            c2 += (p[0]-1)*1000

    if p[0] < 1:
        # 1st sigma must be smaller than second
        if abs(p[1]) > abs(p[2]):
            c2 += abs((((p[1]/p[2])-1)*100)/p[0])

    dif = abs(abs(p[1]-p[2])/((p[1]+p[2])/2.))/10.

    # no negative sigmas
    if p[1]<0 or p[2]<0:
        c1+=abs(min(p[1],p[2]))*100000

    # no negative weight
    if p[0] < 0:
        c2+=abs(p[0]*10000)
    #c2 = (r-1)*f # ok
    #c2 = (r-1)*f

    #err = 0.1
    #return [c1,c2,dif]#,np.log(abs(r))/2.]
    return [c1/20,c2/0.1,c3,c4]#,dif]#,np.log(abs(r))/2.]
    #return [c1/20,c2,c3/10,c4/10]#,dif]#,np.log(abs(r))/2.]

def constrainResScale(p,*args):
    res = ((np.array(p)-1)**3)/1e-4
    global itercount1D
    try: itercount1D += 1
    except: itercount1D = 0
    if itercount1D%2000 == 0:
        print 'constrainResScale',p,abs(res)
    return abs(res)

# p = [fn,wx,wy]
def constrainFactor(p,*args):
    if len(p) == 0:
        return [0]

    # fn between 0 and 1
    c1 = abs(p[0])+abs(1-p[0]) - 1 # =0

    # weigth ratios similar when fn -> 0
    #r = (max(p[1],p[2])/min(p[1],p[2]))
    r = (max(abs(p[1]),abs(p[2]))/min(abs(p[1]),abs(p[2])))
    dif = abs(abs(p[1]-p[2])/((p[1]+p[2])/2.))
    #c2 = ((max(r-1,dif)/1000.)/((p[0]+0.01)**3))/100.
    #c2 = ((max(r-1,dif)/1000.)/((p[0]+0.01)**2))/100.
    c2 = ((max(r-1,dif)/1000.)/((p[0]+0.01)**2))/10.
    #c2 = ((max(r-1,dif)/1000.)/((p[0]+0.01)**2))/1.

    fa = 1-p[1]+0.01

    # negative weight should prefere zero factorability
    # comment out for super Gaussian
    #if p[1] > 1 or p[2] > 1:
    #        fb = p[0]*p[1]*p[2]
    #        c2 += fb*1.
    #err = 0.0001
    #err2 = 0.001

    #err = 0.01
    #err2 = 0.1

    err = 0.005
    err2 = 0.05
    return [c1/err,c2/err2]

# p = [wx,w,s1,s2] # given wx, contrain w,s1,s2 such that w=wx if single Gaussian
def constrainFactorSigmas(p,*args):
    rms = np.sqrt(abs(p[1]*p[2]**2+(1-p[1])*p[3]**2))
    if p[1] > 0.5:
        s = p[2]
    else:
        s = p[3]

    rs = abs(1-(rms/s)+0.001) # small if single Gaussian
    rd = (rs*10)**2*10

    c1 = abs(p[0]-p[1])/rd # p[0]-p[1] is wx wy weight difference

    err = 1.
    return [c1/err]

#########################################
# Velo resolution parametrization
def resNtracks(n,axis='X'):
    # sigma, delta, epsilon
    px = [0.107,-3.0,-0.0038]
    py = [0.106,-2.5,-0.004]
    p = []
    if axis == 'X': p = px
    elif axis == 'Y': p = py
    return (p[0]/(n**(0.5+(p[1]/n**2))))+p[2]

def correctionZ(z,beam=1, axis='X'):
    # R(z) = m + b * z
    m = 0
    p = []
    result = []
    if beam == 1:
        # m, b
        px = [1.06,-0.0019]
        mx = 1.22
        py = [1.06,-0.0018]
        my = 1.23
        if axis == 'X':
            p = px
            m = mx
        elif axis == 'Y':
            p = py
            m = my
        for zi in z:
            if zi >= -1000 and zi < -100:
                result.append(p[0] + zi * p[1])
            elif zi >= -100 and zi <= 500:
                result.append(mx)
            else:
                result.append(1)

    elif beam == 2:
        # m, b
        px1 = [0.83,0.0014]
        py1 = [0.94,0.0012]
        px2 = [0.3,0.0018]
        py2 = [0.2,0.0018]
        if axis == 'X':
            p1 = px1
            p2 = px2
        elif axis == 'Y':
            p1 = py1
            p2 = py2
        for zi in z:
            if zi >= 0 and zi < 700:
                result.append(p1[0] + zi * p1[1])
            elif zi >= 700 and zi <= 1000:
                result.append(p2[0] + zi * p2[1])
            else:
                result.append(1)

    return result

# given N tracks and Z, return resolution
def resolution(n,z,beam=1, axis='X'):
    if beam == 3:
        return resNtracks(n,axis)
    else:
        return resNtracks(n,axis) * correctionZ(z,beam,axis)

def rfBucketTobcid(rfbucket):
    if type(rfbucket) == int or type(rfbucket) == float:
        return int((rfbucket + 9)/10)
    else:
        l = []
        for rfb in rfbucket:
            l.append(int((rfb + 9)/10))
    return l
def bcidToRfBucket(bcid):
    return int((bcid*10)-9)


# detector single interaction PDF
def oneInterationSpectrum(hist,counter=''):
    if counter != '':
        counters = [counter]
    else:
        counters = hist.keys()

    for counter in counters:

        mubb = -np.log(1-(hist[counter]['bb']/float(hist[counter]['bb0']+hist[counter]['bb'])))
        mube = -np.log(1-(hist[counter]['be']/float(hist[counter]['be0']+hist[counter]['be'])))
        mueb = -np.log(1-(hist[counter]['eb']/float(hist[counter]['eb0']+hist[counter]['eb'])))
        #if 'ee' in hist[counter].keys():
        #    muee = -np.log(1-(hist[counter]['ee']/float(hist[counter]['ee0']+hist[counter]['ee'])))
        #else:
        #    muee = 0

        mu = mubb - mube - mueb# + muee
        print 'counter',counter,'mubb mube mueb mu',f(mubb,2),f(mube,2),f(mueb,2),f(mu,2),hist[counter]['bb']

        PbbF=np.fft.rfft(hist[counter]['h']/float(sum(hist[counter]['h'])))
        PbeF=np.fft.rfft(hist[counter]['hbe']/float(sum(hist[counter]['hbe'])))
        PebF=np.fft.rfft(hist[counter]['heb']/float(sum(hist[counter]['heb'])))

        PbkgF = [np.prod(pair) for pair in zip(PbeF, PebF)]

        coff = len(PbbF)/3
        #PbbF[-coff:]=[np.average(PbbF[-coff:])]*coff
        #PbkgF[-coff:]=[np.average(PbkgF[-coff:])]*coff


        I = np.fft.irfft((np.log(PbbF/PbkgF)/mu) + 1)
        InoBkg = np.fft.irfft((np.log(PbbF)/mu) + 1)

        Ibe = np.fft.irfft((np.log(PbeF)/mube) + 1)
        Ieb = np.fft.irfft((np.log(PebF)/mueb) + 1)

        hist[counter]['Ibb'] = I
        hist[counter]['Ibe'] = Ibe
        hist[counter]['Ieb'] = Ieb
        hist[counter]['InoBkg'] = InoBkg

    if counter != '':
        return hist[counter]['Ibb']




##########################
# read from CASTOR
import string,subprocess, shlex

def open_pipe(command):
  split = shlex.split(command)
  p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
  return p.stdout

def castor_load(path):
  command = 'rfcat ' + path
  if path.endswith('gz'):
    command += ' | gunzip'
  elif path.endswith('bz2'):
    command += ' | bunzip2'
  pipe = open_pipe(command)
  return pickle.load(pipe)


##########################
### IO and cuts
##########################

# http://docs.scipy.org/doc/numpy/user/basics.types.html
def read_root2numpy(fname,ignore=[],entries=-1):
    import ROOT
    from root_numpy import tree2array

    f = ROOT.TFile.Open(fname)
    l=f.GetListOfKeys()
    t = ROOT.TTree()
    f.GetObject(l.First().GetName()+"/ntuple", t)

    d = {}
    branches = []
    for branch in t.GetListOfBranches():
        brname = branch.GetName()
        branches.append(brname)

    hasrec = False
    if 'rec_x' in branches: hasrec=True

    # change key names to match old names

    nkey = {
    'run'          :'run',
    'evt'          :'evt',
    'gps'          :'gps',
    'bcid'         :'bcid',
    'bx'           :'bx',
    'rec_nTr'      :'trNum',
    'rec_nTrBw'    :'trNumBackw',
    'rec_x'        :'pvx',
    'rec_y'        :'pvy',
    'rec_z'        :'pvz',
    'rec_1_nTr'    :'trNum.1',
    'rec_1_x'      :'pvx.1',
    'rec_1_y'      :'pvy.1',
    'rec_1_z'      :'pvz.1',
    'rec_2_nTr'    :'trNum.2',
    'rec_2_x'      :'pvx.2',
    'rec_2_y'      :'pvy.2',
    'rec_2_z'      :'pvz.2',
    'rec_lr_1_nTr' :'trNumlr.1',
    'rec_lr_1_x'   :'pvxlr.1',
    'rec_lr_1_y'   :'pvylr.1',
    'rec_lr_1_z'   :'pvzlr.1',
    'rec_lr_2_nTr' :'trNumlr.2',
    'rec_lr_2_x'   :'pvxlr.2',
    'rec_lr_2_y'   :'pvylr.2',
    'rec_lr_2_z'   :'pvzlr.2',
    'hlt_nTr'      :'trNum',
    'hlt_nTrBw'    :'trNumBackw',
    'hlt_x'        :'pvx',
    'hlt_y'        :'pvy',
    'hlt_z'        :'pvz',
    'hlt_1_nTr'    :'trNum.1',
    'hlt_1_x'      :'pvx.1',
    'hlt_1_y'      :'pvy.1',
    'hlt_1_z'      :'pvz.1',
    'hlt_2_nTr'    :'trNum.2',
    'hlt_2_x'      :'pvx.2',
    'hlt_2_y'      :'pvy.2',
    'hlt_2_z'      :'pvz.2',
    'lc_0'         :'lc.0',
    'lc_1'         :'lc.1',
    'lc_2'         :'lc.2',
    'lc_3'         :'lc.3',
    'lc_4'         :'lc.4',
    'lc_6'         :'lc.6',
    'lc_11'        :'lc.11',
    'lc_14'        :'lc.14'
    }

    ntype = {
    'run'          :np.uint32,
    'evt'          :np.uint32,
    'gps'          :np.uint64,
    'bcid'         :np.int16,
    'bx'           :np.uint8,
    'rec_nTr'      :np.uint8,
    'rec_nTrBw'    :np.uint8,
    'rec_x'        :np.float32,
    'rec_y'        :np.float32,
    'rec_z'        :np.float32,
    'rec_1_nTr'    :np.uint8,
    'rec_1_x'      :np.float32,
    'rec_1_y'      :np.float32,
    'rec_1_z'      :np.float32,
    'rec_2_nTr'    :np.uint8,
    'rec_2_x'      :np.float32,
    'rec_2_y'      :np.float32,
    'rec_2_z'      :np.float32,
    'rec_lr_1_nTr' :np.uint8,
    'rec_lr_1_x'   :np.float32,
    'rec_lr_1_y'   :np.float32,
    'rec_lr_1_z'   :np.float32,
    'rec_lr_2_nTr' :np.uint8,
    'rec_lr_2_x'   :np.float32,
    'rec_lr_2_y'   :np.float32,
    'rec_lr_2_z'   :np.float32,
    'hlt_nTr'      :np.uint8,
    'hlt_nTrBw'    :np.uint8,
    'hlt_x'        :np.float32,
    'hlt_y'        :np.float32,
    'hlt_z'        :np.float32,
    'hlt_1_nTr'    :np.uint8,
    'hlt_1_x'      :np.float32,
    'hlt_1_y'      :np.float32,
    'hlt_1_z'      :np.float32,
    'hlt_2_nTr'    :np.uint8,
    'hlt_2_x'      :np.float32,
    'hlt_2_y'      :np.float32,
    'hlt_2_z'      :np.float32,
    'lc_0'         :np.uint16,
    'lc_1'         :np.uint16,
    'lc_2'         :np.uint16,
    'lc_3'         :np.uint16,
    'lc_4'         :np.uint8,
    'lc_6'         :np.uint8,
    'lc_11'        :np.uint8,
    'lc_14'        :np.uint8
    }


    for branch in t.GetListOfBranches():
        brname = branch.GetName()
        if brname in ignore: continue
        if brname not in nkey: continue
        if hasrec and 'hlt_' in brname: continue

        print '-> keep',brname,'->',nkey[brname]
        d[nkey[brname]] = np.array(tree2array(t, branches=[brname],start=0,stop=entries)[brname],dtype=ntype[brname])

    # number of digits in gps time
    if 'gps' in d:
        overd = flist(d['gps'][0], 1)[1]-10
        divider = 10**overd
        print 'gps: divide time by',divider

        d['gps'] /= divider
        d['gps'].astype(np.uint32)

        print 'N events =',len(d['gps'])


    return d


## read ASCII file created by ntupleToTxt.py
# the header words are the dict keys
# each key is a np.array
def readtxt(file, entries=0):
    name,ext = os.path.splitext(file)
    if ext == '.bz2':
        fl = bz2.BZ2File(file, 'rb')
    elif ext == '.gz':
        fl = gzip.open(file, 'rb')
    else:
        fl = open(file, 'r')

    haveheader = False
    d = {}
    
    while not haveheader:
        line = fl.readline()

        if line[0] == '#':
            continue
        
        # find any of those = header line
        # take the 2nd column to find the header
        # the first word might also be present earlier in the header
        # this works for the files
        # - fills2010.txt
        # - 1_13_8_8_8.txt and co.
        # - Fill_XXXX_reduced.txt.bz2
        start_word = ['dd','evt','beam1']
        #if line[0] == '#': line = line[1:] # strip '#' char

        # store data in a dictionary
        # read the header   
        fields = line.strip().split()

        if len(fields) > 1:# and fields[1] in start_word:
            haveheader = True
            print fields
            for field in fields:
               d[field] = array('d') # array of double

    print 'read data...'
    l = 0
    for line in fl:
        #print line
        vals = line.strip().split()
        for i in range(len(vals)):
            d[fields[i]].append(float(vals[i]))
        l += 1
        if l % 500000 == 0: print "read",l,"lines"
        if entries != 0 and l == entries: break

    print l,'entries read'
    
    for field in d:
        #print field
        d[field] = np.array(d[field]) # convert to numpy array

    #print "closing file "+file
    
    fl.close()

    
    #print fields
    
    return d


### store BPM vector into dict
# key 0 == unix time
def readBpmCsv(file):
    print 'read',file

    spl=','
    name,ext = os.path.splitext(file)
    if ext == '.bz2':
        fl = bz2.BZ2File(file, 'rb')
    elif ext == '.gz':
        fl = gzip.open(file, 'rb')
    else:
        fl = open(file, 'r')

    #haveheader = False
    d = {}
    
    # head is composed of 2 lines
    #while not haveheader:
    for l in (0,1,2):
        line = fl.readline()
        if l == 2:
            fields = line.strip(spl).split(spl)
            print 'vector length',len(fields)

            for i in range(len(fields)):
                d[i] = array('d') # array of int
                #d[i].append(int(fields[i].split('.')[0]))
                #print fields[i], len(fields[i])
                d[i].append(float(fields[i]))
        
    l = 0
    for line in fl:
        vals = line.rstrip('\r\n').split(spl)
        
        for i in range(len(vals)):
            d[i].append(float(vals[i]))
                    
        l += 1
        if l % 200000 == 0: print "read",l,"lines"

    
    for field in d:
        d[field] = d[field][:-5]
        d[field] = np.array(d[field]) # convert to numpy array

    d[0] = d[0]/1000 # convert units  ms => s 

    fl.close()

    print l, " entries"
    
    return d


### store FBCT vector into dict
# key 0 == unix time
def readFbctCsv(file):
    print 'read',file

    spl=','
    name,ext = os.path.splitext(file)
    if ext == '.bz2':
        f = bz2.BZ2File(file, 'rb')
    elif ext == '.gz':
        f = gzip.open(file, 'rb')
    else:
        f = open(file, 'r')

    haveheader = False
    d = {}
    
    # head is composed of 1 to 3 lines (or more?) for l in (0,1,2):
    while not haveheader:
    
        line = f.readline()
        if line[0].isdigit():
            haveheader = True
            fields = line.strip(spl).split(spl)
            #print 'vector length',len(fields)

            for i in range(len(fields)):
                d[i] = array('d') # array of int
                #d[i].append(int(fields[i].split('.')[0]))
                #print fields[i], len(fields[i])
                d[i].append(float(fields[i]))
        
    l = 0
    for line in f:
        vals = line.rstrip('\r\n').split(spl)
        
        for i in range(len(vals)):
            d[i].append(float(vals[i]))
                    
        l += 1
        if l % 200000 == 0: print "read",l,"lines"

    
    for field in d:
        d[field] = d[field][:-5]
        d[field] = np.array(d[field]) # convert to numpy array

    d[0] = d[0]/1000 # convert from ms to s

    f.close()

    print l, " entries"
    
    return d

### read CAL_PRE_OFFSETS vectors from DB
# looks like this (fill 1637):
"""
VARIABLE: LHC.BCTDC.A6R4.B1:CAL_PRE_OFFSETS
Timestamp (UNIX Format),Array Values
1300561693317,1.544,1.66133333333333,5.186,22.465
VARIABLE: LHC.BCTDC.A6R4.B2:CAL_PRE_OFFSETS
Timestamp (UNIX Format),Array Values
1300561699421,-0.981666666666666,-2.764,0.091,7.13533333333333
VARIABLE: LHC.BCTDC.B6R4.B1:CAL_PRE_OFFSETS
Timestamp (UNIX Format),Array Values
1300561696365,0.273666666666667,-0.191,3.382,18.991
VARIABLE: LHC.BCTDC.B6R4.B2:CAL_PRE_OFFSETS
Timestamp (UNIX Format),Array Values
1300561702492,0.301,-0.583,3.14066666666667,18.0903333333333
"""
# there is a bug with the LD command (from logginf DB)
# output is not unix time
def readCalPreOffsets(file):
    #print 'read',file

    spl=','
    name,ext = os.path.splitext(file)
    if ext == '.bz2':
        f = bz2.BZ2File(file, 'rb')
    elif ext == '.gz':
        f = gzip.open(file, 'rb')
    else:
        f = open(file, 'r')

    #haveheader = False
    d = {}
    
    var = ''
    for line in f:
        if 'VARIABLE' in line:
            var = line.split()[1]
            continue
            
        if 'Timestamp' in line:
            continue
        
        vals = line.rstrip('\r\n').split(spl)

        d[var] = []
        for i in range(len(vals)):
            if i == 0:
                d[var].append(timeDateEpoch(vals[i]))
            else:
                d[var].append(float(vals[i]))
                    
    for field in d:
        d[field] = np.array(d[field]) # convert to numpy array
        d[field][0] = d[field][0]/1000 # convert from ms to s

    f.close()

    return d

### store DB values (timber) into dict
# key 0 == unix time
def readDbCsv(file,lastifempty=False,type='l'):
    #print 'read',file

    spl=','
    name,ext = os.path.splitext(file)
    if ext == '.bz2':
        f = bz2.BZ2File(file, 'rb')
    elif ext == '.gz':
        f = gzip.open(file, 'rb')
    else:
        f = open(file, 'r')

    #haveheader = False
    d = {}
    
    # head is composed of 1 line = keys
    line = f.readline()
    fields = line.rstrip('\r\n').strip(spl).split(spl)
    fields[0] = 'date'
    print fields,len(fields)

    for field in fields:
        d[field] = array('d') # array of int or dec
    
        
    l = 0
    for line in f:
        vals = line.rstrip('\r\n').split(spl)
        #print vals
        
        for i in range(len(fields)):
            if len(vals[i]) == 0: # no value
                if lastifempty and len(d[fields[i]]) > 0:
                    d[fields[i]].append(d[fields[i]][-1])
                else:
                    d[fields[i]].append(0)
            else:
                #d[fields[i]].append(int(vals[i]))
                d[fields[i]].append(float(vals[i]))
                    
        l += 1
        if l % 200000 == 0: print "read",l,"lines"

    
    for field in d:
        d[field] = np.array(d[field]) # convert to numpy array

    d['date'] = d['date']/1000 # convert from ms to s

    f.close()

    print l, " entries"
    
    return d


# detect steps in continuous array
# correspond to a jump in data (e.g. injection, new current)

def detectSteps(fld,diff):
    dsteps = []


    for i in range(5,len(fld)-1):
        mavg = np.average(fld[i-5:i]) # average of last 5 points
        #diff = 1e12 # use 20 for linear3; 100 otherwise

        excl = range(i-10,i+10)
        #if abs(fld[i+1] - mavg) > diff \ # positive and negative step
        if fld[i+1] - mavg > diff \
                and len([val for val in dsteps if val in excl])==0:

            #print 'step',i,fld[i+1],fld[i],fld[i-1],mavg,fields[0]

            dsteps.append(i+1)


    return dsteps
    
## read ASCII file from LPC
# there is no header
# time = in seconds since january 1, 2010, 00:00:00  (CET).
# @param file the file name
# @param headers list of string to define the each column
# @param entries the number of entries to read (0 == all)
def readlpc2010(file, fields=[''],entries=0):
    name,ext = os.path.splitext(file)
    if ext == '.bz2':
        f = bz2.BZ2File(file, 'rb')
    elif ext == '.gz':
        f = gzip.open(file, 'rb')
    else:
        f = open(file, 'r')

    d = {}
    for h in fields:
        d[h] = array('d')

    l = 0
    warned = False
    for line in f:
        #print line
        vals = line.strip().split()
        if len(fields) != len(vals) and not warned:
            print "WARNING: ",str(len(fields))," fields expected, but found",str(len(vals)),"file",file
            warned = True
        
        if len(vals) == 0: # due to empty line
            continue
        
        for i in range(len(fields)):
            d[fields[i]].append(float(vals[i]))
        l += 1
        if l % 200000 == 0: print "read",l,"lines"
        if entries != 0 and l == entries: break

    for field in d:
       d[field] = np.array(d[field]) # convert to numpy array

    f.close()

    #print l, " entries"
    #print fields
    
    return d

## read ASCII file from LPC
# there is no header
# time = in seconds since january 1, 2010, 00:00:00  (CET).
# @param file the file name
# @param headers list of string to define the each column
# @param entries the number of entries to read (0 == all)
def readlpc(file, fields=[''],entries=0,spl=None):
    if type(file) == types.FileType:
        f = file
    # probably an open zip archive
    elif type(file).__name__ == 'instance' or type(file).__name__ == 'ExFileObject':
        f = file
    else:
        name,ext = os.path.splitext(file)
        if ext == '.bz2':
            f = bz2.BZ2File(file, 'rb')
        elif ext == '.gz':
            f = gzip.open(file, 'rb')
        else:
            f = open(file, 'r')

    d = {}
    for h in fields:
        d[h] = array('d')

    l = 0
    warned = False
    for line in f:
        #print line
        if line[0] == '#':
            continue
        
        vals = line.strip().split(spl)
        if len(fields) != len(vals) and not warned:
            print "WARNING: ",str(len(fields))," fields expected, but found",str(len(vals)),"file",file,'line',l
            warned = True
        
        if len(vals) == 0: # due to empty line
            continue
        
        for i in range(len(fields)):
            d[fields[i]].append(float(vals[i]))
        l += 1
        if l % 200000 == 0: print "read",l,"lines"
        if entries != 0 and l == entries: break

    for field in d:
       d[field] = np.array(d[field]) # convert to numpy array

    f.close()

    #print l, " entries"
    #print fields
    
    return d


# FBCT time average (rebinning)
def tavgfbct(d,dt=300,t1=0):
    keys = d.keys()
    nd = {}
    # first copy extra dict fields (e.g. x_avg)
    #for k in d.keys():
    #    nd[k] = d[k][:]
    for k in keys:
        nd[k] = array('l')

    if t1 == 0:
        t1 = d[0][0]

    t2 = t1 + dt

    j = 0 # start index of time interval

    while t2 < d[0][-1]:

        idx = np.where((d[0]>=t1)&(d[0]<t2))
        if len(idx[0]) > 0:

            for k in keys:
                if k == 0:
                    nd[k].append(int((t1+t2)/2.))
                else:
                    nd[k].append(int(np.average(d[k][idx])))

        t1 = t2
        t2 += dt

    for field in nd:
        nd[field] = np.array(nd[field]) # convert to numpy array

    return nd

def tavgdcct(d,dt=300,t1=0):
    keys = d.keys()
    nd = {}
    # first copy extra dict fields (e.g. x_avg)
    #for k in d.keys():
    #    nd[k] = d[k][:]
    for k in keys:
        nd[k] = array('l')

    if t1 == 0:
        t1 = d['date'][0]

    t2 = t1 + dt

    while t2 < d['date'][-1]:

        idx = np.where((d['date']>=t1)&(d['date']<t2))[0]

        if len(idx) > 0:

            for k in keys:
                if k == 'date':
                    nd[k].append(int((t1+t2)/2.))
                else:
                    idxz = np.where(d[k][idx]!=0)[0]
                    #idx = np.where((d['date']>=t1)&(d['date']<t2)&(d[k]!=0))[0]
                    #print k,len(idx),d[k][idx]
                    if len(idxz) > 0:
                        nd[k].append(int(np.average(d[k][idx][idxz])))
                    else:
                        nd[k].append(0)

        t1 = t2
        t2 += dt

    for field in nd:
        nd[field] = np.array(nd[field]) # convert to numpy array

    return nd

## read ASCII file from LPC
# there is a header, header list read from file
# @param file the file name
def readlpcfills(file):
    name,ext = os.path.splitext(file)
    if ext == '.bz2':
        f = bz2.BZ2File(file, 'rb')
    elif ext == '.gz':
        f = gzip.open(file, 'rb')
    else:
        f = open(file, 'r')

    d = {}
    fields = []

    haveheader = False

    for line in f:
        if not haveheader:
            if line[:5] != '#fill' and line[:4] != '# nr':
                continue
            else:
                fields = line.strip('#').strip().split()
                if fields[2] == 'mm': fields[2] = 'm'
                for h in fields:
                    d[h] = array('d')
                haveheader = True
                continue
        elif line[0] != '#': # read dato into dict
            vals = line.strip().split()
            for i in range(len(vals)):
                d[fields[i]].append(float(vals[i]))

    for field in d:
       d[field] = np.array(d[field]) # convert to numpy array

    f.close()

    #print fields
    
    return d

def readtab(file, entries=0):
    return readcsv(file, entries)
    
## read PVSS CSV file
# lastifempty=True will use last known value if no value found for timestamp
def readcsv(file, entries=0, headerpos=0,isfloat=True,split=',', lastifempty=False,verbose=True):
    #spl='\t'
    # headerpos is the line position
    spl=split
    name,ext = os.path.splitext(file)
    if ext == '.bz2':
        f = bz2.BZ2File(file, 'rb')
    elif ext == '.gz':
        f = gzip.open(file, 'rb')
    else:
        f = open(file, 'r')

    haveheader = False
    d = {}
    
    # head is composed of 2 lines
    #while not haveheader:
    firstchar = spl
    l = 0
    while l <= headerpos or firstchar == spl:
        line = f.readline()
        #print l,line
        firstchar = line[0]
        if firstchar == spl or l < headerpos: # take second line for the dict keys
            l += 1
            continue

        # store data in a dictionary
        # read the header
        line = line.rstrip('\r\n')
        if line[0] == spl:
            line = 'date'+line

        fields = line.strip(spl).split(spl)
        if 'imestamp' in fields[0]:
            fields[0] = 'date'
        
        for field in fields:
            if 'date' in field or not isfloat:
                d[field] = []
                print field,'as string array'
            else:
                d[field] = array('d') # array of double
        
        l += 1
        #print l,firstchar,line

        #if headerpos == 0 and line[0] == spl:
        #    l = 0

    print 'header:',l
    #print d.keys(),len(d.keys())
    #print fields,len(fields)
    l = 0
    for line in f:
        vals = line.rstrip('\r\n').split(spl)
        #print vals
        
        for i in range(len(fields)):
            

            if 'date' in fields[i] or not isfloat: # first column is a string (the date)
                d[fields[i]].append(vals[i])
            else:
                if len(vals[i]) == 0: # insert zero if no value
                    if lastifempty and len(d[fields[i]])>0:
                        d[fields[i]].append(d[fields[i]][-1])
                    else:
                        d[fields[i]].append(0)
                else:
                    d[fields[i]].append(float(vals[i]))

            #print fields[i], d[fields[i]][-1]
                    
        l += 1
        if l % 200000 == 0: print "read",l,"lines"
        if entries != 0 and l == entries: break

    for field in d:
       d[field] = np.array(d[field]) # convert to numpy array

    f.close()

    if verbose:
        print l, " entries"
        print fields
    
    return d

def readpvss(file):
    f = open(file, 'r')

    #haveheader = False
    d = {}
    
    # no header
    # first 6 column: 2011 11 15 19 57 55 

    d['gps'] = array('d')
    d['val'] = array('d')
    

    #print d.keys(),len(d.keys())
    #print fields,len(fields)
    l = 0
    for line in f:
        vals = line.rstrip('\r\n').split()

        timeTuple = (int(vals[0]),int(vals[1]),int(vals[2]),int(vals[3]),int(vals[4]),int(vals[5]))
        tgps = timeTupleGps(timeTuple,toffset)

        d['gps'].append(tgps)
        d['val'].append(float(vals[6]))
                            
        l += 1
        if l % 20000 == 0: print "read",l,"lines"

    for field in d:
       d[field] = np.array(d[field]) # convert to numpy array

    f.close()

    print l, " entries"
    
    return d

## read simple CSV file without header, specify fields
def readcsvsimple(file,fields):
    spl=','
    name,ext = os.path.splitext(file)
    if ext == '.bz2':
        f = bz2.BZ2File(file, 'rbU')
    elif ext == '.gz':
        f = gzip.open(file, 'rb')
    else:
        f = open(file, 'r')

    d = {}
    for field in fields:
        d[field] = array('d') # array of double

    for line in f:
        vals = line.rstrip('\r\n').split(spl)
        
        for i in range(len(fields)):
            d[fields[i]].append(float(vals[i]))

    for field in d:
       d[field] = np.array(d[field]) # convert to numpy array

    f.close()

    print fields
    
    return d

## read DCCT MD data (csv of ADC bins = raw data)
def readMdData(file, entries=0, split=','):
    #spl='\t'
    spl=split
    name,ext = os.path.splitext(file)
    if ext == '.bz2':
        f = bz2.BZ2File(file, 'rb')
    elif ext == '.gz':
        f = gzip.open(file, 'rb')
    else:
        f = open(file, 'r')

    #haveheader = False
    d = {}
    
    # head is composed of 3 lines
    #while not haveheader:
    for l in range(3):
        line = f.readline()
        if l != 2:
            continue

        # store data in a dictionary
        # use header info as dict key
        # this is the last line of the header
        #line = line.rstrip('\r\n')

        fields = line.replace(',','').strip('# ').split(' ')[0:5]
        
        for field in fields:
            d[field] = array('d') # array of double
        
        break

    #print d.keys(),len(d.keys())
    #print fields,len(fields)
    l = 0
    for line in f:
        vals = line.rstrip('\r\n').split(spl)
        
        for i in range(len(fields)):
            d[fields[i]].append(float(vals[i]))
                    
        l += 1
        if l % 200000 == 0: print "read",l,"lines"
        if entries != 0 and l == entries: break

    for field in d:
       d[field] = np.array(d[field]) # convert to numpy array

    f.close()

    #print l, " entries, fields:",fields
    
    return d

def readcsvxls(file, entries=0, skipl=0, split=','):
    #spl='\t'
    spl=split
    name,ext = os.path.splitext(file)
    if ext == '.bz2':
        f = bz2.BZ2File(file, 'rb')
    elif ext == '.gz':
        f = gzip.open(file, 'rb')
    else:
        f = open(file, 'rU')

    #haveheader = False
    d = {}
    
    # head is composed of 2 lines
    #while not haveheader:
    l=-1
    while l<skipl:
        line = f.readline() # read header as keys
        l+=1
        print l,line

    # store data in a dictionary
    # read the header
    line = line.rstrip('\r\n')
    if line[0] == spl:
        line = 'date'+line

    fields = line.strip(spl).split(spl)
    
    for field in fields:
        if field == 'date':
            d[field] = []
        else:
            d[field] = array('d') # array of double

    #print d.keys(),len(d.keys())
    #print fields,len(fields)
    for line in f:
        vals = line.rstrip('\r\n').split(spl)
        print vals
        for i in range(len(vals)):

            if len(vals[i]) == 0: # insert zero if no value
                d[fields[i]].append(0)
            else:
                d[fields[i]].append(float(vals[i]))

    for field in d:
       d[field] = np.array(d[field]) # convert to numpy array

    f.close()

    print fields
    
    return d

def filterOutsiders(a):
    idx = np.where(a!=0)
    avg = np.average(a[idx])
    for i in range(len(a)):
        if a[i] > 15*avg:
            a[i] = 0
    return a

def normalize(a,idx=None):
    maxval = max(a)
    if idx is not None:
        maxval = max(a[idx])
    
    nstr,dec,bool,decades = flist(maxval,1)
    if maxval/10**decades > 1.2:
        decades += 1
    print maxval,nstr,decades
    for i in range(len(a)):
        a[i] = a[i] / 10**decades

    return a,decades
    
    
if __name__ == "__main__":
    import sys
    cut = None
    if len(sys.argv) > 2: cut = sys.argv[2]
    readtxt(sys.argv[1], cut)

def idxAll(d):
    idx = range(len(d[d.keys()[0]]))
    return idx

def idxbx(d,bx,idx=None):
    if 'bx' in d.keys():
        bxtkey = 'bx'
    if 'ODIN_bx' in d.keys():
        bxtkey = 'ODIN_bx'

    newidx = np.where(d[bxtkey] == bx)[0]
    
    if idx is not None:
        nidx=np.intersect1d(idx,newidx)
        return nidx
    else:
        return newidx

# routBitBeamGas = 1 -> beam gas
# routBitBeamGas = 0 -> no beam gas
def idxBitBeamGas(d,bg=1,idx=None):
    if idx is None:
        idx = range(len(d['run']))
    if 'routBitBeamGas' not in d.keys():
        print "no routBitBeamGas bit in data"
        return idx
    newidx = []
    for i in idx:
        if d['routBitBeamGas'][i] == bg:
            newidx.append(i)

    return newidx

# remove double events WARNING only useful is some conditions (e.g. MC analysis)
# keep only the first
def idxUniqEvents(d,idx=None):
    if idx is None:
        idx = range(len(d['run']))
    newidx = []
    evts = []
    for i in idx:
        if d['evt'][i] not in evts:
            newidx.append(i)
            evts.append(d['evt'][i])
    return newidx

# separate in beam 1 and beam 2 according to fwd/bwd ratio
# Beam 1 PU can be 0   trBkw<<trFwd
# Beam 2 SPD can be 0  trFwd<<trBkw
def idxBeamSeparate(d,idx=None):
    if idx is None:
        idx = range(len(d['run']))
    newidx1 = [] # beam 1
    newidx2 = [] # beam 2
    for i in idx:
        trBkw = d['trNumBackw'][i]
        trFwd = d['trNumForw'][i]
        if trBkw == 0: newidx1.append(i)
        elif trFwd == 0: newidx2.append(i)
        else:
            if trBkw/float(trFwd) < 0.1: newidx1.append(i)
            elif trFwd/float(trBkw) < 0.1: newidx2.append(i)
            
    return newidx1, newidx2

# separate in beam 1 and beam 2 according to fwd/bwd ratio 
# WARNING allow 1 track in other direction
def idxBeamBG1(d,beam,idx=None):
    if beam == 3:
        return idx

    if beam == 1:
        newidx = np.where(d['trNumBackw'] <= 1)[0]
    if beam == 2:
        if 'trNumForw' in d:
            newidx = np.where(d['trNumForw'] <= 1)[0]
        else:
            newidx = np.where(d['trNum']-d['trNumBackw'] <= 1)[0]
    
    if idx is not None:
        nidx=np.intersect1d(idx,newidx)
        return nidx
    else:
        return newidx

# separate in beam 1 and beam 2 according to fwd/bwd ratio
def idxBeamBG(d,beam,idx=None):
    if beam == 3:
        return idx

    if beam == 1:
        newidx = np.where(d['trNumBackw'] == 0)[0]
    if beam == 2:
        if 'trNumForw' in d:
            newidx = np.where(d['trNumForw'] == 0)[0]
        else:
            newidx = np.where(d['trNum']-d['trNumBackw'] == 0)[0]
    
    if idx is not None:
        nidx=np.intersect1d(idx,newidx)
        return nidx
    else:
        return newidx

# select beam spot BB only
def idxBB(d,idx=None):

    newidx = np.where((d['trNum']-d['trNumBackw'] >= 1)&(d['trNumBackw'] >= 1)&(abs(d['pvz'])<=300))[0]

    if idx is not None:
        nidx=np.intersect1d(idx,newidx)
        return nidx
    else:
        return newidx

def idxZrange(d,r=[],idx=None,MC=False):
    if not MC: pvz = 'pvz'
    else: pvz = 'mcpvz'
    

    if len(r) < 2:
        print 'WARNING bad range format',t
        return idx

    newidx = np.where((d[pvz]>=r[0])&(d[pvz]<r[1]))[0]
    
    if idx is None:
        return newidx
    else:
        nidx=np.intersect1d(idx,newidx)
        return nidx


# maximal error allowed on x,y,z
def idxZmaxError(d,x,y,z,idx=None):
    if idx is None:
        idx = range(len(d['run']))
    newidx = []
    for i in idx:
        if (d['pvxe'][i]<=x and d['pvye'][i]<=y and d['pvze'][i]<=z):
            newidx.append(i)
    return newidx

## return bad vertices
# MC - reco > 10 mm
def idxBadVetricesMC(d,idx=None):
    if 'err' not in d.keys():
        print "not a MC data set or no err key"
        return idx
    if idx is None:
        idx = range(len(d['run']))
    newidx = []
    maxr = 0.1
    for i in idx:
        r = np.sqrt((d['pvy'][i]-d['mcpvy'][i])**2 + (d['pvx'][i]-d['mcpvx'][i])**2)
        if r > maxr:
            newidx.append(i)
        elif abs(d['pvz'][i]-d['mcpvz'][i]) > 10*maxr:
            newidx.append(i)
    return newidx

## separate good and bad vertices 
def idxGoodBadVetricesMC(d,idx=None):
    if 'err' not in d.keys():
        print "not a MC data set or no err key"
        return idx
    if idx is None:
        idx = range(len(d['run']))
    idxg = []
    idxb = []

    dxr = 0.12
    dzr = 2
    for i in idx:
        good = True
        r = np.sqrt((d['pvy'][i]-d['mcpvy'][i])**2 + (d['pvx'][i]-d['mcpvx'][i])**2)
        if r > dxr + abs(d['pvz'][i])/1000.:
            good = False
        elif abs(d['pvz'][i]-d['mcpvz'][i]) > dzr+abs(d['pvz'][i])/100.:
            good = False
        if good:
            idxg.append(i)
        else:
            idxb.append(i)
    return idxg, idxb

# gpsend is either end time or duration in sec
def idxTimeRange(d,trange=[],idx=None):
    
    if len(trange) < 2:
        return idx
    
    gpsbegin = trange[0]
    gpsend = trange[1]
    if gpsend < gpsbegin: gpsend = gpsbegin + gpsend
    
    newidx = np.where((d['gps']>=gpsbegin)&(d['gps']<gpsend))[0]

    if idx is not None:
        nidx=np.intersect1d(idx,newidx)
        return nidx
    else:
        return newidx
    

def idxMaxRadius(d,rmax,idx=None):
    if idx is None:
        idx = range(len(d['run']))
    
    newidx = np.where(np.sqrt(d['pvx']**2 + d['pvy']**2) < rmax)[0]
    nidx=np.intersect1d(idx,newidx)

    return nidx

def idxHaveSplitVertex(d,idx=None):
    if idx is None:
        idx = range(len(d['run']))
        
    newidx = np.where((d['pvz.1']!=0)&(d['pvz.2']!=0))[0]
    nidx=np.intersect1d(idx,newidx)

    return newidx

def idxRuns(d,runs=[],idx=None):
    newidx = []
    if idx is None:
        idx = range(len(d['run']))

    for r in runs:
        idxr = np.where(d['run']==r)[0]
        newidx = np.concatenate((newidx,idxr))
        
    if len(runs) == 0:
        return idx.astype(np.int64)

    idx=np.intersect1d(idx,newidx)
    return idx.astype(np.int64)

def idxRunsExclude(d,runs=[],idx=None):
    if idx is None:
        idx = range(len(d['run']))
    newidx = []
    if len(runs) == 0:
        return idx
    for i in idx:
        if (d['run'][i] not in runs):
            newidx.append(i)

    return newidx

# select bunches
def idxBunches(d,bunches=[],idx=None):
    cidx = []
    for bunch in bunches:
        newidx = np.where(d['bcid'] == bunch)[0]
        cidx = np.concatenate((cidx,newidx))

    if idx is not None:
        nidx=np.intersect1d(idx,cidx)
        return nidx.astype(np.int64)
    else:
        return cidx.astype(np.int64)

# select bunches
def idxBunch(d,bunch,idx=None):
    
    newidx = np.where(d['bcid'] == bunch)[0]
    if idx is not None:
        nidx=np.intersect1d(idx,newidx)
        return nidx
    else:
        return newidx


# select events by event ID
def idxEvents(d,events=[],idx=None):
    if idx is None:
        idx = range(len(d['evt']))
    newidx = []
    if len(bunches) == 0:
        return idx
    for i in idx:
        if (d['evt'][i] in bunches):
            newidx.append(i)

    return newidx

# minimum number of reconstructed vertices (typically 0 or 1)
def idxMinVertices(d,v=1,idx=None):
    if 'nPVsInEvent' not in d:
        return idx
    if idx is None:
        idx = range(len(d['run']))
    newidx = []
    for i in idx:
        if (d['nPVsInEvent'][i] >= v):
            newidx.append(i)
    return newidx

# minimum tracks in reconstructed vertex (for any event/vertex)
def idxMinTracks(d,mintr=4,idx=None):
    
    newidx = np.where(d['trNum'] >= mintr)[0]
    if idx is not None:
        nidx=np.intersect1d(idx,newidx)
        return nidx
    else:
        return newidx


# minimum tracks in reconstructed vertex for luminous region only
def idxMinTracksPP(d,mintr=10,lum=(-250,250),idx=None):
    if idx is None:
        idx = range(len(d['run']))
    newidx = []
    for i in idx:
        if d['bx'][i] != 3:
            newidx.append(i)
        elif d['pvz'][i] < lum[0] or d['pvz'][i] > lum[1]:
            newidx.append(i)
        elif (d['trNum'][i] >= mintr):
            newidx.append(i)

    return newidx

# minimum of 3d velo (incl. long) tracks (all of them)
# NOT per vertex
def idxMin3Dtracks(d,tr=0,idx=None):
    if 'nTracks3D' not in d:
        return idx
    if idx is not None:
        idx = range(len(d['run']))
    newidx = []
    for i in idx:
        if (d['nTracks3D'][i] >= tr):
            newidx.append(i)

    return newidx

def idxMax3Dtracks(d,tr=100,idx=None):
    if 'nTracks3D' not in d:
        return idx
    if idx is None:
        idx = range(len(d['run']))
    newidx = []
    for i in idx:
        if (d['nTracks3D'][i] < tr):
            newidx.append(i)

    return newidx

# min/maxof 3d velo tracks (all of them)
def idxMinMax3Dtracks(d,trmin=0,trmax=100,idx=None):
    if 'nTracks3D' not in d:
        return idx
    if idx is None:
        idx = range(len(d['run']))
    idxmin = idxMin3Dtracks(d,trmin,idx)
    return idxMax3Dtracks(d,trmax,idxmin)

def idxResolRange(d,begin,end,beam=1,axis='X',idx=None):
    if idx is None:
        idx = range(len(d['run']))
    newidx = []
    r = resolution(d['trNum'],d['pvz'],beam=beam,axis=axis)
    for i in idx:
        if r[i]>=begin and r[i]<end:
            newidx.append(i)

    return newidx

# wrapper to select a beam data (e.g. b1, bb, etc.)
def idxBeam(d,bx=3,zmax=1000,rmax=1,idx=None,runs=[],mintr=3):
    if idx is None:
        idx = range(len(d['run']))
    idxrun = idxRuns(d,runs=runs,idx=idx)
    idxradius = idxMaxRadius(d,rmax=rmax,idx=idxrun)
    idxtracks = idxMinTracks(d,mintr=mintr,idx=idxradius)
    newidx = []
    for i in idxtracks:
        if d['bx'][i]==bx and abs(d['pvz'][i])<zmax:
            newidx.append(i)
    
    return newidx

# select luminous region only bb
def idxLum(d,zdist=250,rmax=1,idx=None,runs=[],mintr=10):
    if idx is None:
        idx = range(len(d['run']))
    idxrun = idxRuns(d,runs=runs,idx=idx)
    idxradius = idxMaxRadius(d,rmax=rmax,idx=idxrun)
    idxtracks = idxMinTracks(d,mintr=mintr,idx=idxradius)
    idxzrange = idxZrange(d,[-zdist,zdist],idx=idxtracks)
    newidx = []
    for i in idxzrange:
        if d['bx'][i]==3:
            newidx.append(i)
    
    return newidx


# select events from a beam
# take bx = beam
# for pp exclude luminous region
def idxSelectBeam(d,beam=1,idx=None):
    if idx is None:
        idx = range(len(d['run']))

    newidx = []
    for i in idx:
        if d['bx'][i]==beam:
            newidx.append(i)
        else:
            if beam==1 and d['pvz'][i] < -300:
                newidx.append(i)
            if beam==2 and d['pvz'][i] > 300:
                newidx.append(i)
    
    return newidx

# keep only unique events
# double events are expected to be continuous
def idxUniqEvents(d,idx=None):
    print "filter unique vertices out of total:",len(idx)
    if idx is None:
        idx = range(len(d['evt']))
    idxu = [idx[:1]]
    for i in range(len(idx)):
        if i>0 and d['evt'][idx[i]] != d['evt'][idx[i-1]]:
            idxu.append(idx[i])

    return idxu

# keep only unique events
# double events are expected to be continuous
def idxFilterBeamGas(d,beam=0,idx=None):
    if idx is None:
        idx = range(len(d['evt']))
    idxgood = []
    idxbad = []
    maxd = 3
    stat=[0,0,0]
    for i in idx:
        b = beam
        # only use reconstructed vertices
        if (d['nPVsInEvent'][i] == 0):
            continue
        dist = np.sqrt(d['pvy'][i]**2 + d['pvx'][i]**2)
        if dist > maxd:
            idxbad.append(i)
            stat[0]+=1
            continue

        ntr  = d['trNum'][i]
        nfwd = d['trNumForw'][i]
        nbkw = d['trNumBackw'][i]
        # detect beam if not known
        bxt = int(d['bx'][i])

        if b == 0:
            if bxt != 1 or bxt != 2:
                # use fwd/bkw tracks to find out which beam it is
                if nfwd > nbkw:
                    b = 1
                else:
                    b = 2
            else:
                b = bxt
        
        # use fwd/bwd ratio
        ratio = (nfwd-nbkw)/float(ntr)
        if b == 1:
            #if d['pvz'][i] > -300:
            #    if ratio > 0.95:
            #        idxgood.append(i)
            #if d['pvz'][i] < -300: # tracks ratio has no sence behind the PU
            #    idxgood.append(i)
            
            if ratio > 0.98:
                idxgood.append(i)
            else:
                idxbad.append(i)
        if b == 2:
            if d['pvz'][i] > 990: # far away for sure BG
                idxgood.append(i)
            elif ratio < -0.98:  # closer check assymetry
                idxgood.append(i)
            else:
                idxbad.append(i)
    #print stat,len(idx),len(idxgood),len(idxbad)
    return idxgood,idxbad


##########################
### misc tools
##########################

def float_to_decimal(f):
    # http://docs.python.org/library/decimal.html#decimal-faq
    #"Convert a floating point number to a Decimal with no loss of information"
    n, d = f.as_integer_ratio()
    numerator, denominator = decimal.Decimal(n), decimal.Decimal(d)
    ctx = decimal.Context(prec=60)
    result = ctx.divide(numerator, denominator)
    while ctx.flags[decimal.Inexact]:
        ctx.flags[decimal.Inexact] = False
        ctx.prec *= 2
        result = ctx.divide(numerator, denominator)
    return result 

def flist(number, sigfig):
    # http://stackoverflow.com/questions/2663612/nicely-representing-a-floating-point-number-in-python/2663623#2663623
    if sigfig <= 0: sigfig = 1
    try:
        d=decimal.Decimal(number)
    except TypeError:
        d=float_to_decimal(float(number))
    sign,digits,exponent=d.as_tuple()
    if len(digits) < sigfig:
        digits = list(digits)
        digits.extend([0] * (sigfig - len(digits)))    
    shift=d.adjusted()
    result=int(''.join(map(str,digits[:sigfig])))
    # Round the result
    if len(digits)>sigfig and digits[sigfig]>=5: result+=1
    result=list(str(result))
    # Rounding can change the length of result
    # If so, adjust shift
    shift+=len(result)-sigfig
    # reset len of result to sigfig
    result=result[:sigfig]
    #print "len(res) ",len(result)," shift ",shift
    #print "result ",result ," sigfig ",sigfig," shift ",shift," digits ",digits
    sround=0
    dec = False
    
    if shift >= sigfig-1:
        # Tack more zeros on the end
        result+=['0']*(shift-sigfig+1)
        sround = len(result)
    elif 0<=shift:
        # Place the decimal point in between digits
        result.insert(shift+1,'.')
        sround = len(result)-1
        #dec = True
    else:
        # Tack zeros on the front
        assert(shift<0)
        result=['0.']+['0']*(-shift-1)+result
        sround = len(result)-1
        dec = True
    if sign:
        result.insert(0,'-')

    return ''.join(result),sround,dec,shift

def f(number, sigfig):
    return flist(number,sigfig)[0]

def fe(number,error, sigfig):
    #print 'fe',number,error, sigfig
    err, srounde, dece, shifte = flist(error,sigfig)
    n, sroundn, decn, shiftn = flist(number,sigfig)

    #print n,err,sroundn,srounde,shiftn,shifte,dece,decn
    
    if dece and decn:
        #n, sroundn, decn, shift = flist(number,srounde)
        #print number,srounde,shiftn,srounde+shiftn+1
        n, sroundn, decn, shift = flist(number,srounde+shiftn+1)
        #n, sroundn, decn, shift = flist(number,srounde+shiftn)
        
    elif dece and not decn:
        n, sroundn, decn, shift = flist(number,sroundn+srounde)
        #n, sroundn, decn, shift = flist(number,sroundn+srounde-1)
    elif not dece and not decn:
        n, sroundn, decn, shift = flist(number,sroundn-srounde+1)
        #n, sroundn, decn, shift = flist(number,max(sroundn-srounde+1,sigfig))
    
    #print number,sroundn,srounde,shift,dece,decn
    return n,err

# format error into string
# return a string val +- err
def fes(number,error, sigfig):
    n,e = fe(number,error,sigfig)
    return '$'+n+'\pm'+e+'$'

#-------------------------
# http://depts.washington.edu/clawpack/users/claw/python/pyclaw/plotters/colormaps.py
def make_colormap(colors):
#-------------------------
    """
    Define a new color map based on values specified in the dictionary
    colors, where colors[z] is the color that value z should be mapped to,
    with linear interpolation between the given values of z.

    The z values (dictionary keys) are real numbers and the values
    colors[z] can be either an RGB list, e.g. [1,0,0] for red, or an
    html hex string, e.g. "#ff0000" for red.
    """

    from matplotlib.colors import LinearSegmentedColormap, ColorConverter
    from numpy import sort
    
    z = sort(colors.keys())
    n = len(z)
    z1 = min(z)
    zn = max(z)
    x0 = (z - z1) / (zn - z1)
    
    CC = ColorConverter()
    R = []
    G = []
    B = []
    for i in range(n):
        #i'th color at level z[i]:
        Ci = colors[z[i]]      
        if type(Ci) == str:
            # a hex string of form '#ff0000' for example (for red)
            RGB = CC.to_rgb(Ci)
        else:
            # assume it's an RGB triple already:
            RGB = Ci
        R.append(RGB[0])
        G.append(RGB[1])
        B.append(RGB[2])

    cmap_dict = {}
    cmap_dict['red'] = [(x0[i],R[i],R[i]) for i in range(len(R))]
    cmap_dict['green'] = [(x0[i],G[i],G[i]) for i in range(len(G))]
    cmap_dict['blue'] = [(x0[i],B[i],B[i]) for i in range(len(B))]
    mymap = LinearSegmentedColormap('mymap',cmap_dict)
    return mymap


###########
# create data and bin to plot a histogram outline with pyplab.plot
# adapted from
# http://www.scipy.org/Cookbook/Matplotlib/UnfilledHistograms?action=AttachFile&do=get&target=histNofill.py

# takes numpy histogram and its bins
def histOutline(histIn, binsIn):
    """
    Make a histogram that can be plotted with plot() so that
    the histogram just has the outline rather than bars as it
    usually does.

    Example Usage:
    binsIn = numpy.arange(0, 1, 0.1)
    angle = pylab.rand(50)

    (data, bins) = histOutline(angle, binsIn)
    plot(bins, data, 'k-', linewidth=2)

    """

    stepSize = binsIn[1] - binsIn[0]

    bins = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    data = np.zeros(len(binsIn)*2 + 2, dtype=np.float)    
    for bb in range(len(binsIn)):
        bins[2*bb + 1] = binsIn[bb]
        bins[2*bb + 2] = binsIn[bb] + stepSize
        if bb < len(histIn):
            data[2*bb + 1] = histIn[bb]
            data[2*bb + 2] = histIn[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0
    
    return data, bins
    #return bins, data


# http://www.peterbe.com/plog/uniqafiers-benchmark
def uniqify(seq, idfun=None): # Alex Martelli ******* order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker not in seen:
            seen[marker] = 1
            result.append(item)
            
    return result

mkers = [ '+' , '*' , ',' , '.',
    '1' , '2' , '3' , '4',
    '<' , '>' , 'D' , 'H',
    '^' , '_' , 'd' , 'h',
    'o' , 'p' , 's' , 'v',
    'x' , '|']

toffset = 1259625600
slowearth = 16. # wether or not to use 15-16 s offset due the slower earth rotation
# gps is time givent by the event
# returns time tuple (y,m,d,h,m,s)
def gpsTuple(gps):
# odin gps in microseconds, gps time is 13sec ahead of UTC
# information from http://www.csgnetwork.com/timegpsdispcalc.html says 15sec ahead
# Plamen tuple has offset and is in seconds
# gpss = gps+toffset-slowearth
 gpss = gps-slowearth
 timeTuple = time.localtime(gpss)
# (tm_year, tm_mon, tm_day, tm_hour, tm_min, tm_sec, tm_wday, tm_yday, tm_isdst)
 sec   = "%02d" % (timeTuple[5])
 minute = "%02d" % (timeTuple[4])
 txt = str(timeTuple[2])+'.'+str(timeTuple[1])+'. '+str(timeTuple[0])+'  '+str(timeTuple[3])+':'+minute+':'+sec
# retun YYYY MM DD time
 #return str(timeTuple[0]),str(timeTuple[1]),str(timeTuple[2]),str(timeTuple[3])+':'+minute+':'+sec
 #return txt
 return timeTuple[0],timeTuple[1],timeTuple[2],timeTuple[3],timeTuple[4],timeTuple[5]

def timeTupleGps(timeTuple, offset=0):
    #print timeTuple
    t = datetime(timeTuple[0],timeTuple[1],timeTuple[2],timeTuple[3],timeTuple[4],timeTuple[5])
    tunix = mktime(t.timetuple())
    tgps = tunix-toffset+slowearth+offset
    return tgps

def timeOdinToGps(gps):
# odin gps in microseconds, gps time is 13sec ahead of UTC
# information from http://www.csgnetwork.com/timegpsdispcalc.html says 15sec ahead
# Plamen tuple has offset and is in seconds
    return gps+toffset-slowearth

def timeGpsToOdin(gps):
    return gps-toffset+slowearth

def timeDateGps(dateString, offset=0):
    # now = '2011-12-19 12:50:00.0'
    # timeDateGps(now)
    # 64669815.0
    #
    # in format '2010/10/15 13:18:28.434'
    # or format '2010-10-15 07:22:42.000'

    if dateString[4] == '/':
        t = datetime.strptime(dateString, '%Y/%m/%d %H:%M:%S.%f')
    elif dateString[4] == '-':
        t = datetime.strptime(dateString, '%Y-%m-%d %H:%M:%S.%f')
#    if UTC:
#        tunix = calendar.timegm(t.timetuple())
#    else:
    tunix = mktime(t.timetuple())
    tgps = tunix-toffset+slowearth+offset
    return tgps

def timeDateEpoch(dateString, utc=False):
    # in format '2010/10/15 13:18:28.434'
    # or format '2010-10-15 07:22:42.000'

    if dateString[4] == '/':
        t = time.strptime(dateString, '%Y/%m/%d %H:%M:%S.%f')
    elif dateString[4] == '-':
        t = time.strptime(dateString, '%Y-%m-%d %H:%M:%S.%f')

    if utc:
        tunix = calendar.timegm(t)
    else:
        tunix = mktime(t)
    

    return tunix

# convert gps time to numDate
def GpsNum(gps,offset=0):
    gpss = gps+offset-slowearth
    timeTuple = time.localtime(gpss)
    t = datetime(timeTuple[0],timeTuple[1],timeTuple[2],timeTuple[3],timeTuple[4],timeTuple[5])
    return mpl.dates.date2num(t)
    
def GpsNums(gps,offset=0):
    numdates = []
    for j in range(len(gps)):
        numdates.append(GpsNum(gps[j],offset))

    return np.array(numdates)

# matplotlib numerical date
# http://matplotlib.sourceforge.net/api/dates_api.html?highlight=date#matplotlib.dates
def timeDateNum(dateString, offset=0):
    # in format '2010/10/15 13:18:28.434'
    # or format '2010-10-15 07:22:42.000'

    #ndates = epoch2num(D.d['time'])

    if dateString[4] == '/':
        t = datetime.strptime(dateString, '%Y/%m/%d %H:%M:%S.%f')
    elif dateString[4] == '-':
        t = datetime.strptime(dateString, '%Y-%m-%d %H:%M:%S.%f')
    elif dateString[2] == '/':
        t = datetime.strptime(dateString, '%d/%m/%Y %H:%M:%S')
    elif len(dateString) == 12: # i.e. format 201108300910
        t = datetime.strptime(dateString, '%Y%m%d%H%M')
    else:
        print 'wrong date format', dateString,type(dateString)
        
        return 0
    
    return mpl.dates.date2num(t)

def timeDateNums(dateString):
    # dateString iterable (list or array)
    #print len(dateString)
    numdates = []
    for j in range(len(dateString)):
        numdates.append(timeDateNum(dateString[j]))
    
    return np.array(numdates)

def toBunchSlot(rfbucket):
    return int(rfbucket+9 /10)

# return charges for a given current (LHC only)
#### convert to charges per bin
# 1 A = 1C/s = I
# 1 e = 1.602e-19 C
# f = 11245 1/s
# C = A * s = 1/1.6e-19 e
# Ctot = I/f * 1/e 

def currentToCharges(I):
    flhc = 11245.5 # Hz
    el = 1.602176487E-19 # C
    return (I/flhc) * 1/el

def chargesToCurrent(c):
    flhc = 11245.5 # Hz
    el = 1.602176487E-19 # C
    return flhc * el * c
    

    
# use interpolated function outside boundaries
# from http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-a-an-extrapolated-result-beyond-the-input-rang
"""
x = arange(0,10)
y = exp(-x/3.0)
f_i = interp1d(x, y)
f_x = extrap1d(f_i)

print f_x([9,10])

[ 0.04978707  0.03009069]

"""
# use e.g. si = interpolate.interp1d(t, s, kind='linear')
def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(map(pointwise, array(xs)))

    return ufunclike


def extrap1dsame(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        #print x,xs[0],interpolator(x)
        if x < xs[0]:
            return ys[0]
        elif x > xs[-1]:
            return ys[-1]
        else:
            return interpolator(x)

    def ufunclike(xs):
        #return sp.array(map(pointwise, sp.array(xs)))
        return pointwise(xs)

    return ufunclike

def runToFill(run):
    if run in [154501,154502,154503,154504]: return 3850
    if run in range(161105,161123): return 4266
    if run in range(161169,161204): return 4269

