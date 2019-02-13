# 
# Bryan Kaiser
# 2/12/19

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math as ma
#import scipy
import h5py
import functions as fn

figure_path = "./figures/"
data_path = "/home/bryan/data/dlm/data/"
filename = 'microdata.h5'
alpha_level = 0.1

# in sess.run(train_op,feed_dict={x: x_data3, y: y_data3}), what sizes can x_data be?
# add histograms, statistics
# set stdev with data?

# =============================================================================
# get data for MDN

f = h5py.File( data_path + filename , 'r')
N2 = f['N2'][:]
CT = f['CT'][:]
SA = f['SA'][:]
eps = f['eps'][:]
z = f['z'][:]
Np = np.shape(eps)[0]
#print(Np)

# data needs to be shape (N,1):
Nchunk = 10000
Noffset = 1000000
x_data = np.zeros([Nchunk,1])
x_data[:,0] = z[Noffset:Nchunk+Noffset]
y_data = np.zeros([Nchunk,1])
y_data[:,0] = np.log10(eps[Noffset:Nchunk+Noffset])

Nchunk = 2000
Noffset = 1100000
x_test = np.zeros([Nchunk,1])
x_test[:,0] = z[int(2*Nchunk)+Noffset:int(3*Nchunk)+Noffset]
y_test = np.zeros([Nchunk,1])
y_test[:,0] = np.log10(eps[int(2*Nchunk)+Noffset:int(3*Nchunk)+Noffset])

"""
x_data2 = np.zeros([Nchunk,4])
x_data2[:,0] = z[Noffset:Nchunk+Noffset]
x_data2[:,1] = CT[Noffset:Nchunk+Noffset]
x_data2[:,2] = SA[Noffset:Nchunk+Noffset]
x_data2[:,4] = N2[Noffset:Nchunk+Noffset]
"""

#x_data = x_data/np.amin(x_data)-np.mean(x_data/np.amin(x_data))
#y_data = y_data/np.amin(y_data)-np.mean(y_data/np.amin(y_data))
#x_test = x_test/np.amin(x_test)-np.mean(x_test/np.amin(x_test))
#y_test = y_test/np.amin(y_test)-np.mean(y_test/np.amin(y_test))

print(np.shape(x_data))
print(np.shape(y_data))
print(np.shape(x_test))
print(np.shape(y_test))


plotname = figure_path +'micro_training_data.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(y_data,x_data,'bo',alpha=alpha_level)
plt.ylabel(r"z",fontsize=13)
plt.xlabel(r"log$_{10}(\varepsilon)$",fontsize=13)
plt.title(r"training data",fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);
plt.show()

plotname = figure_path +'micro_test_data.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(y_test,x_test,'ro',alpha=alpha_level)
plt.ylabel(r"z",fontsize=13)
plt.xlabel(r"log$_{10}(\varepsilon)$",fontsize=13)
plt.title(r"testing data",fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);
plt.show()


# =============================================================================
# A simple dense layer NN:

# 1) place holders as variables that will eventually hold data, to do symbolic 
# computations on the graph later:
x = tf.placeholder(dtype=tf.float32, shape=[None,1])
y = tf.placeholder(dtype=tf.float32, shape=[None,1])

# 2) construct/initialize a neural network one-hidden layer and 20 units:
NHIDDEN = 32
W = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=1.0, dtype=tf.float32))
b = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=1.0, dtype=tf.float32))
W_out = tf.Variable(tf.random_normal([NHIDDEN,1], stddev=1.0, dtype=tf.float32))
b_out = tf.Variable(tf.random_normal([1,1], stddev=1.0, dtype=tf.float32))
hidden_layer = tf.nn.tanh(tf.matmul(x, W) + b)
y_out = tf.matmul(hidden_layer,W_out) + b_out


# 3) define a loss function as the sum of square error of the output vs the data 
lossfunc = tf.nn.l2_loss(y_out-y) # the loss function is what we iteratively minimize


# 4) define a training operator that will tell TensorFlow to minimize the loss 
# function later. Here we choose the RMSProp gradient descent optimisation method:
train_op = tf.train.RMSPropOptimizer(learning_rate=0.1, decay=0.8).minimize(lossfunc)


# 5) A session object must be defined to use Tensorflow. The run command initializes 
# all variables, and where the computation graph will also be built inside TensorFlow.
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init) 
# tf.initialize_all_variables()) # initialize_all_variables is outdated
# Use `tf.global_variables_initializer` instead.


# 6) predictions on the training data (fit)
NEPOCH = 10000
for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: x_data, y: y_data})
#x_test = np.zeros([Nchunk,1])
#x_test[:,0] = np.linspace(np.amin(x_test),np.amax(x_test),num=Nchunk)
y_pred = sess.run(y_out,feed_dict={x: x_data})

plotname = figure_path +'micro_training_prediction_nn.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(y_data,x_data,'bo',alpha=alpha_level,label=r"data")
plt.plot(y_pred,x_data,'ko',alpha=alpha_level,label=r"NN")
plt.ylabel(r"z",fontsize=13)
plt.xlabel(r"log$_{10}(\varepsilon)$",fontsize=13)
plt.title(r"training data",fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);


# 7) predictions on the testing data (fit)
y_pred2 = sess.run(y_out,feed_dict={x: x_test})

plotname = figure_path +'micro_test_prediction_nn.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(y_test,x_test,'ro',alpha=alpha_level,label=r"data")
plt.plot(y_pred2,x_test,'ko',alpha=alpha_level,label=r"NN")
plt.ylabel(r"z",fontsize=13)
plt.xlabel(r"log$_{10}(\varepsilon)$",fontsize=13)
plt.title(r"testing data",fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);



sess.close() # close the session afterwards to free resources


# =============================================================================
# A mixture density network


# construct the MDN:
NHIDDEN = 24
STDEV = 0.5
KMIX = 32 # number of mixtures
NOUT = KMIX * 3 # pi, mu, stdev

x = tf.placeholder(dtype=tf.float32, shape=[None,1], name="x") 
y = tf.placeholder(dtype=tf.float32, shape=[None,1], name="y")

Wh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))
bh = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=STDEV, dtype=tf.float32))

Wo = tf.Variable(tf.random_normal([NHIDDEN,NOUT], stddev=STDEV, dtype=tf.float32))
bo = tf.Variable(tf.random_normal([1,NOUT], stddev=STDEV, dtype=tf.float32))

hidden_layer = tf.nn.tanh(tf.matmul(x, Wh) + bh)
output = tf.matmul(hidden_layer,Wo) + bo

out_pi, out_sigma, out_mu = fn.get_mixture_coeff(output,KMIX)

lossfunc = fn.get_lossfunc(out_pi, out_sigma, out_mu, y)
train_op = tf.train.AdamOptimizer().minimize(lossfunc)

# training:
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

NEPOCH = 20000
loss = np.zeros(NEPOCH) # store the training progress here.
for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: x_data, y: y_data})
  loss[i] = sess.run(lossfunc, feed_dict={x: x_data, y: y_data})

# testing:
out_pi_test, out_sigma_test, out_mu_test = sess.run(fn.get_mixture_coeff(output,KMIX), feed_dict={x: x_test})
out_pi_train, out_sigma_train, out_mu_train = sess.run(fn.get_mixture_coeff(output,KMIX), feed_dict={x: x_data})
# out_pi_test, out_sigma_test, out_mu_test = each are size (Nchunk,Kmix)

y_pred_test = fn.generate_ensemble( out_pi_test, out_mu_test, out_sigma_test, x_test )
y_pred_train = fn.generate_ensemble( out_pi_train, out_mu_train, out_sigma_train, x_data )
# y_pred = size(Nchunk,10?)

"""
print(np.shape(y_test))
print(np.shape(x_test))
print(np.shape(y_pred))
print(out_mu_test)
print(out_pi_test)
print(out_sigma_test)
print(np.shape(out_mu_test))
print(np.shape(out_pi_test))
print(np.shape(out_sigma_test))
"""

plotname = figure_path +'micro_test_prediction_mdn.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(y_test,x_test,'ro',alpha=alpha_level,label=r"data")
plt.plot(y_pred_test[:,0],x_test,'ko',alpha=alpha_level,label=r"MDN")
plt.plot(y_pred_test[:,1],x_test,'bo',alpha=alpha_level,label=r"MDN")
plt.plot(y_pred_test[:,2],x_test,'co',alpha=alpha_level,label=r"MDN")
plt.ylabel(r"z",fontsize=13)
plt.xlabel(r"log$_{10}(\varepsilon)$",fontsize=13)
plt.title(r"testing data",fontsize=13)
#plt.axis([-12.,-5.,-4500.,100.])
plt.savefig(plotname,format="png"); plt.close(fig);


plotname = figure_path +'micro_training_prediction_mdn.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(y_data,x_data,'bo',alpha=alpha_level,label=r"data")
plt.plot(y_pred_train[:,0],x_data,'ko',alpha=alpha_level,label=r"MDN")
plt.plot(y_pred_train[:,1],x_data,'bo',alpha=alpha_level,label=r"MDN")
plt.plot(y_pred_train[:,2],x_data,'co',alpha=alpha_level,label=r"MDN")
plt.ylabel(r"z",fontsize=13)
plt.xlabel(r"log$_{10}(\varepsilon)$",fontsize=13)
plt.title(r"training data",fontsize=13)
#plt.axis([-12.,-5.,-4500.,100.])
plt.savefig(plotname,format="png"); plt.close(fig);






sess.close()



