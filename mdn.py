# An example of a simple single dense layer neural network learning algorithm
# Bryan Kaiser
# 2/11/19

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
#import scipy
#import h5py

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

plotname = figure_path +'predictions.png' 
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

plotname = figure_path +'predictions2.png' 
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






























"""

# =============================================================================
# functions

def hidden_layer( y, W, W0, w, w0 , function_flag ):
 # a single hidden layer NN with a single network output, Z
 Nm = np.shape(W0)[0]
 z = np.zeros([Nm]) 
 for j in range(0,Nm): # for each hidden unit 
   z[j] = sum(y[:,0]*W[j,:]) + W0[j,0]

 if function_flag == 'ReLU':
   Z = sum(ReLU(z[:])*w[:,0]) + w0 # network output
 return Z

def ReLU( self ):
 f = np.zeros(np.shape(self)[0])
 for k in range(0,np.shape(self)[0]):
  if self[k] > 0.:
    f[k] = self[k]
 return f

def output_layer( y, w, w0 , function_flag ):
 # a single hidden layer NN with a single network output, Z
 N = np.shape(w)[0] # number of inputs to the single output
 z = sum(np.multiply(y,w)) + w0
 #if function_flag == 'ReLU':
 #  Z = sum(ReLU(z[:])*w[:,0]) + w0 # network output
 return z

def hinge( y, z ):
  loss = max(0.,1-y*z)
  return loss

# =============================================================================
# make the data

Nd = 1000 # number of inputs
x = np.float32(np.random.uniform(-10.5, 10.5, (1, Nd))).T
r = np.float32(np.random.normal(size=(Nd,1)))
y = np.float32(np.sin(0.75*x)*7.0+x*0.5+r*1.0)

plotname = figure_path +'data.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(x,y,'ro',alpha=0.3)
plt.xlabel(r"x",fontsize=13)
plt.ylabel(r"y(x)",fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);
plt.show()

# =============================================================================
# simple neural network with no hidden layers and 20 nodes, w/o Tensorflow

w = np.float32(np.random.normal(size=(Nd,1))) 
w0 = np.float32(np.random.normal(size=(1))) # one bias for hidden layer output


# if you sum over x and there is one z output, how does each x,y pair have a unique z?


# iterate over training examples in random order and update based on the loss: 
# the loss comes from plugging in the x value into the w,w0 to predict y 



# for every x,y pair get a z 
z = output_layer( x, w, w0 ) # one output for an input vector
print(z)
# which y pairs with z?

# output weights, w, w0: update using the SGD method for the single layer model for a linear classifier


# decision boundary at x*w+w0

# =============================================================================
# simple neural network one-hidden layer and 20 nodes, w/o Tensorflow

Nm = 20 # number of hidden layers

# initial weights:
W = np.float32(np.random.normal(size=(Nm,Nd))) 
W0 = np.float32(np.random.normal(size=(Nm,1))) # one bias for each NN input
w = np.float32(np.random.normal(size=(Nm,1))) 
w0 = np.float32(np.random.normal(size=(1))) # one bias for hidden layer output

Z = hidden_layer( y, W, W0, w, w0 , 'ReLU' )
print(Z)

# output weights, w, w0: update using the SGD method for the single layer model for a linear classifier


# hidden weights, W, W0:
# select a random input
# choose weights via minimized loss
# repeat.

# x and y above are x_data

x = tf.placeholder(dtype=tf.float32, shape=[None,1])
y = tf.placeholder(dtype=tf.float32, shape=[None,1])


NHIDDEN = 20
W = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=1.0, dtype=tf.float32))
b = tf.Variable(tf.random_normal([1,NHIDDEN], stddev=1.0, dtype=tf.float32))

W_out = tf.Variable(tf.random_normal([NHIDDEN,1], stddev=1.0, dtype=tf.float32))
b_out = tf.Variable(tf.random_normal([1,1], stddev=1.0, dtype=tf.float32))

hidden_layer = tf.nn.tanh(tf.matmul(x, W) + b)
y_out = tf.matmul(hidden_layer,W_out) + b_out


lossfunc = tf.nn.l2_loss(y_out-y);

"""

