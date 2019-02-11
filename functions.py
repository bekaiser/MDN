

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
# functions

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
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print('error with sampling ensemble')
  return -1


def generate_ensemble(out_pi, out_mu, out_sigma, x_test , M = 10):
  NTEST = x_test.size
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
