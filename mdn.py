# An example of a simple single dense layer neural network learning algorithm
# Bryan Kaiser
# 2/11/19

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math as ma
#import scipy
#import h5py
import functions as fn

figure_path = "./figures/"


# =============================================================================
# make data sets

Nd = 1000 # number of inputs

# data set 1:

x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, Nd))).T
r_data = np.float32(np.random.normal(size=(Nd,1)))
y_data = np.float32(np.sin(0.75*x_data)*7.0+x_data*0.5+r_data*1.0)

plotname = figure_path +'data.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(x_data,y_data,'ro',alpha=0.3)
plt.xlabel(r"x",fontsize=13)
plt.ylabel(r"y(x)",fontsize=13)
plt.title(r"N = %i" %(Nd),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);
plt.show()

# data set 2:

temp_data = x_data
x_data2 = y_data
y_data2 = temp_data

plotname = figure_path +'data2.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(x_data2,y_data2,'ro',alpha=0.3)
plt.xlabel(r"x",fontsize=13)
plt.ylabel(r"y(x)",fontsize=13)
plt.title(r"N = %i" %(Nd),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);
plt.show()

# data set 3:

NSAMPLE = 2500
y_data3 = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE))).T
r_data3 = np.float32(np.random.normal(size=(NSAMPLE,1))) # random noise
x_data3 = np.float32(np.sin(0.75*y_data3)*7.0+y_data3*0.5+r_data3*1.0)

plotname = figure_path +'data3.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(x_data3,y_data3,'ro',alpha=0.3)
plt.xlabel(r"x",fontsize=13)
plt.ylabel(r"y(x)",fontsize=13)
plt.title(r"N = %i" %(NSAMPLE),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);
plt.show()


# =============================================================================
# A simple dense layer NN:

# 1) place holders as variables that will eventually hold data, to do symbolic 
# computations on the graph later:
x = tf.placeholder(dtype=tf.float32, shape=[None,1])
y = tf.placeholder(dtype=tf.float32, shape=[None,1])


# 2) construct/initialize a neural network one-hidden layer and 20 units:
NHIDDEN = 20
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


# 6) run gradient descent for 1000 times to minimise the loss function with the 
# data fed in via a dictionary. After the below is run, the weight and bias parameters
# will be auto stored in their respective tf.Variable objects.
NEPOCH = 1000
for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: x_data, y: y_data})


# 7) make predictions on data set 1:
x_test = np.float32(np.arange(-10.5,10.5,0.1))
#print(np.shape(x_test))
x_test = x_test.reshape(x_test.size,1) # changes dimensions from (N,) to (N,1)
#print(np.shape(x_test))
y_test = sess.run(y_out,feed_dict={x: x_test})

plotname = figure_path +'predictions_nn.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(x_data,y_data,'ro',alpha=0.3,label=r"data")
plt.plot(x_test,y_test,'bo',alpha=0.3,label=r"predictions")
plt.xlabel(r"x",fontsize=13)
plt.ylabel(r"y(x)",fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);
plt.show()

sess.close() # close the session afterwards to free resources 


# 8) make predictions on data set 2:
# the neural network trained to fit only to the square mean y(x) 
# of the data fails on the inverted data set (non-unique trend 
# of y)
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init) 
NEPOCH = 1000
for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: x_data2, y: y_data2})
x_test = np.float32(np.arange(-10.5,10.5,0.1))
x_test = x_test.reshape(x_test.size,1) 
y_test = sess.run(y_out,feed_dict={x: x_test})

plotname = figure_path +'predictions_nn2.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(x_data2,y_data2,'ro',alpha=0.3,label=r"data")
plt.plot(x_test,y_test,'bo',alpha=0.3,label=r"predictions")
plt.xlabel(r"x",fontsize=13)
plt.ylabel(r"y(x)",fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);
plt.show()

sess.close()



# =============================================================================
# A mixture density network


# construct the MDN:
NHIDDEN = 24
STDEV = 0.5
KMIX = 24 # number of mixtures
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

NEPOCH = 10000
loss = np.zeros(NEPOCH) # store the training progress here.
for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: x_data3, y: y_data3})
  loss[i] = sess.run(lossfunc, feed_dict={x: x_data3, y: y_data3})


# testing:
x_test = np.float32(np.arange(-15,15,0.1))
NTEST = x_test.size
x_test = x_test.reshape(NTEST,1) # needs to be a matrix, not a vector

out_pi_test, out_sigma_test, out_mu_test = sess.run(fn.get_mixture_coeff(output,KMIX), feed_dict={x: x_test})

y_test = fn.generate_ensemble( out_pi_test, out_mu_test, out_sigma_test, x_test )


plotname = figure_path +'predictions_mdn.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(x_data3,y_data3,'ro',alpha=0.3,label=r"data")
plt.plot(x_test,y_test,'bo',alpha=0.3,label=r"predictions")
plt.xlabel(r"x",fontsize=13)
plt.ylabel(r"y(x)",fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);
plt.show()

sess.close()


