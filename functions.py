

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math as ma
#import scipy
#import h5py

figure_path = "./figures/"


# =============================================================================
# MDN functions

def get_mixture_coeff(output,KMIX):
  out_pi = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_sigma = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_mu = tf.placeholder(dtype=tf.float32, shape=[None,KMIX], name="mixparam")
  out_pi, out_sigma, out_mu = tf.split(output, num_or_size_splits=3, axis=1)
  max_pi = tf.reduce_max(out_pi, 1, keepdims=True)
  out_pi = tf.subtract(out_pi, max_pi)
  out_pi = tf.exp(out_pi)
  normalize_pi = tf.reciprocal(tf.reduce_sum(out_pi, 1, keepdims=True))
  out_pi = tf.multiply(normalize_pi, out_pi)
  out_sigma = tf.exp(out_sigma)
  return out_pi, out_sigma, out_mu


def tf_normal(y, mu, sigma):
  result = tf.subtract(y, mu)
  result = tf.multiply(result, tf.reciprocal(sigma))
  result = -tf.square(result)/2
  return tf.multiply(tf.exp(result),tf.reciprocal(sigma))/(ma.sqrt(2*ma.pi))


def get_lossfunc(out_pi, out_sigma, out_mu, y):
  result = tf_normal(y, out_mu, out_sigma)
  result = tf.multiply(result, out_pi)
  result = tf.reduce_sum(result, 1, keepdims=True)
  result = -tf.log(result)
  return tf.reduce_mean(result)


def get_pi_idx(x, pdf):
  N = pdf.size
  #print(pdf.size)
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print('error with sampling ensemble')
  return -1


def generate_ensemble(out_pi, out_mu, out_sigma, x_test , M ):
  NTEST = np.shape(x_test)[0] #x_test.size
  #print(NTEST)
  result = np.random.rand(NTEST, M) # initially random [0, 1]
  rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
  mu = 0
  std = 0
  idx = 0

  # transforms result into random ensembles
  for j in range(0, M):
    for i in range(0, NTEST):
      idx = get_pi_idx(result[i, j], out_pi[i])
      mu = out_mu[i, idx]
      std = out_sigma[i, idx]
      result[i, j] = mu + rn[i, j]*std
  return result


# =============================================================================
# output processing functions

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    """
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    """

    if window_len<3:
        return x

    """
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    """

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


# =============================================================================
# hydrography data processing functions

#import os
from netCDF4 import Dataset
import gsw 

def get_hydro(my_file):
 f = Dataset(my_file, mode='r')
 
 eps = f.variables['EPSILON'][:]
 eps = remove_bad_eps( eps )
 lat = f.variables['LATITUDE'][:]
 lon = f.variables['LONGITUDE'][:]
 p = f.variables['PRESSURE'][:]
 SP = f.variables['PSAL'][:]
 T = f.variables['TEMPERATURE'][:]

 z = gsw.z_from_p(p,lat) # m
 SA = gsw.SA_from_SP(SP,p,lon,lat) #  g/kg, absolute salinity
 CT = gsw.CT_from_t(SA,T,p) # C, conservative temperature

 SA = remove_bad_SA( SA )
 CT = remove_bad_CT( CT )

 [N2_mid, p_mid] = gsw.Nsquared(SA,CT,p,lat)
 z_mid = gsw.z_from_p(p_mid,lat)

 N2 = interp_to_edges( N2_mid , z , z_mid , 4)
 N2 = np.append(np.append(N2,[np.nan]),[np.nan])
 N2 = remove_bad_N2( N2 )

 return N2, SA, CT, eps, z

def remove_bad_eps( eps ):
 Neps = np.shape(eps)[0]
 for k in range(0,Neps):
  if abs(eps[k]) >= 1e-2:
    eps[k] = np.nan
  if eps[k] < 0.:
    eps[k] = np.nan
 return eps

def remove_bad_SA( SA ):
 Nsa = np.shape(SA)[0]
 for k in range(0,Nsa):
  if abs(SA[k]) >= 50.:
    SA[k] = np.nan
  if SA[k] < 0.:
    SA[k] = np.nan
 return SA

def remove_bad_CT( CT ):
 Nct = np.shape(CT)[0]
 for k in range(0,Nct):
  if abs(CT[k]) >= 50.:
    CT[k] = np.nan
  if CT[k] <= -50.:
    CT[k] = np.nan
 return CT

def remove_bad_N2( N2 ):
 Nn2 = np.shape(N2)[0]
 for k in range(0,Nn2):
  if abs(N2[k]) >= 1e-1:
    N2[k] = np.nan
  if N2[k] <= -1e-1:
    N2[k] = np.nan
 return N2

def interp_to_centers( self_edge , zc , ze ):
 Nc = np.shape(zc)[0]
 Ne = np.shape(ze)[0]
 self_center = np.zeros([Nc])
 c = np.transpose(weights(zc[0],ze[0:4],4))
 self_center[0] = np.dot( c[0,:] , self_edge[0:4] )
 c = np.transpose(weights(zc[Nc-1] , ze[Ne-5:Ne-1],4))
 self_center[Nc-1] = np.dot( c[0,:] , self_edge[0:4] )
 for j in range(1,Nc-1):
  c = np.transpose(weights(zc[j] , ze[j-1:j+3],0))
  #print(np.shape(c))
  #print(ze[j-1:j+2])
  self_center[j] = np.dot( c[0,:] , self_edge[j-1:j+3] )
  #self_center[j] = fnbg( zc[j] , ze , self , 4 , 0 )
 return self_center

def weights(z,x,m):
# From Bengt Fornbergs (1998) SIAM Review paper.
#  	Input Parameters
#	z location where approximations are to be accurate,
#	x(0:nd) grid point locations, found in x(0:n)
#	n one less than total number of grid points; n must
#	not exceed the parameter nd below,
#	nd dimension of x- and c-arrays in calling program
#	x(0:nd) and c(0:nd,0:m), respectively,
#	m highest derivative for which weights are sought,
#	Output Parameter
#	c(0:nd,0:m) weights at grid locations x(0:n) for derivatives
#	of order 0:m, found in c(0:n,0:m)
#      	dimension x(0:nd),c(0:nd,0:m)

  n = np.shape(x)[0]-1
  c = np.zeros([n+1,m+1])
  c1 = 1.0
  c4 = x[0]-z
  for k in range(0,m+1):  
    for j in range(0,n+1): 
      c[j,k] = 0.0
  c[0,0] = 1.0
  for i in range(0,n+1):
    mn = min(i,m)
    c2 = 1.0
    c5 = c4
    c4 = x[i]-z
    for j in range(0,i):
      c3 = x[i]-x[j]
      c2 = c2*c3
      if (j == i-1):
        for k in range(mn,0,-1): 
          c[i,k] = c1*(k*c[i-1,k-1]-c5*c[i-1,k])/c2
      c[i,0] = -c1*c5*c[i-1,0]/c2
      for k in range(mn,0,-1):
        c[j,k] = (c4*c[j,k]-k*c[j,k-1])/c3
      c[j,0] = c4*c[j,0]/c3
    c1 = c2
  return c

def interp_to_edges( self_center , ze , zc , flag ):
 # self is centers
 Ne = np.shape(ze)[0] # edge number
 Nc = np.shape(zc)[0] # center number
 # e1 c1 e2 c2 e3 c3 e4 c4 e5 c5 e6 c6 e7
 #    in  o in o  in o  in  o in o  in
 self_edge = np.zeros([Nc-1])
 
 if flag == 6:
   for j in range(2,Nc-3):
     c = np.transpose(weights(ze[j],zc[j-2:j+4],0))
     self_edge[j] = np.dot( c[0,:] , self_center[j-2:j+4] )
   c = np.transpose(weights(ze[1],zc[0:6],0))
   self_edge[0] = np.dot( c[0,:] , self_center[0:6] )
   c = np.transpose(weights(ze[2],zc[0:6],0))
   self_edge[1] = np.dot( c[0,:] , self_center[0:6] )
   c = np.transpose(weights(ze[Ne-1],zc[Ne-7:Ne-1],0))
   self_edge[Nc-2] = np.dot( c[0,:] , self_center[Ne-7:Ne-1] )
   c = np.transpose(weights(ze[Ne-2],zc[Ne-7:Ne-1],0))
   self_edge[Nc-3] = np.dot( c[0,:] , self_center[Ne-7:Ne-1] )

 if flag == 4:
   for j in range(1,Nc-2):
     c = np.transpose(weights(ze[j],zc[j-1:j+3],0))
     self_edge[j] = np.dot( c[0,:] , self_center[j-1:j+3] )
   c = np.transpose(weights(ze[1],zc[0:4],0))
   self_edge[0] = np.dot( c[0,:] , self_center[0:4] )
   c = np.transpose(weights(ze[Ne-1],zc[Ne-5:Ne-1],0))
   self_edge[Nc-2] = np.dot( c[0,:] , self_center[Ne-5:Ne-1] )

 if flag == 2:
   for j in range(1,Nc-2):
     c = np.transpose(weights(ze[j],zc[j:j+2],0))
     self_edge[j] = np.dot( c[0,:] , self_center[j:j+2] )
   c = np.transpose(weights(ze[1],zc[0:2],0))
   self_edge[0] = np.dot( c[0,:] , self_center[0:2] )
   c = np.transpose(weights(ze[Nc-2],zc[Ne-3:Ne-1],0))
   self_edge[Nc-2] = np.dot( c[0,:] , self_center[Ne-3:Ne-1] )

 return self_edge
