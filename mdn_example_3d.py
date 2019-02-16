# Simple examples of dense layer neural networks and MDNs
# Bryan Kaiser
# 2/16/19

import matplotlib
matplotlib.use('Agg') # set non-interactive backend for PNGs, called before .pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D # note the capitalization
import tensorflow as tf
import numpy as np
import math as ma
import functions as fn

figure_path = "./figures/"
alpha_level = 0.2


# make more functions ?
# add regressions ?

# =============================================================================
# make data set 1 

NSAMPLE1 = 4000 # number of inputs

x_data = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE1))).T
r_data = np.float32(np.random.normal(size=(NSAMPLE1,1)))
x_data2 = np.float32(np.random.uniform(-10.5, 10.5, (1, NSAMPLE1))).T
r_data2 = np.float32(np.random.normal(size=(NSAMPLE1,1)))
y_data = np.float32(np.sin(0.75*x_data+0.75*x_data2)*7.0+(x_data+x_data2)*0.5+(r_data+r_data2)*0.5)
X_DATA = np.hstack((x_data,x_data2))
"""
# tube:
y_data = np.float32(np.zeros([NSAMPLE1,1]))
y_data[0:int(NSAMPLE1/2),0] = np.float32( - ((10.5)**2. - (x_data[0:int(NSAMPLE1/2),0] - 0.5*r_data[0:int(NSAMPLE1/2),0])**2.  )**(1./2.) - x_data2[0:int(NSAMPLE1/2),0]) 
y_data[int(NSAMPLE1/2):NSAMPLE1,0] = np.float32( ((10.5)**2. - (x_data[int(NSAMPLE1/2):NSAMPLE1,0] - 0.5*r_data[int(NSAMPLE1/2):NSAMPLE1,0])**2. )**(1./2.) - x_data2[int(NSAMPLE1/2):NSAMPLE1,0]) 
"""

plotname = figure_path +'training_data_3d.png' 
fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
zdata = y_data
ydata = x_data2
xdata = x_data
ax.scatter3D(xdata, ydata, zdata, zdir='z', alpha=0.2, s=20, c='k') 
ax.set_xlabel(r"x$_1$",fontsize=13)
ax.set_ylabel(r"x$_2$",fontsize=13)
ax.set_zlabel(r"y",fontsize=13)
ax.view_init(elev=20., azim=130)
ax.set_xlim3d(-15, 15)
ax.set_ylim3d(-15, 15)
ax.set_zlim3d(-20,20)
plt.title(r"training data 1, $N_{samples}=$%i" %(NSAMPLE1),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);
"""
plotname = figure_path +'training_data_2d_1.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(x_data,y_data,'ko',alpha=alpha_level)
plt.xlabel(r"x$_1$",fontsize=13)
plt.ylabel(r"y(x)",fontsize=13)
plt.title(r"$N_{samples}=$%i" %(NSAMPLE1),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

plotname = figure_path +'training_data_2d_2.png' 
fig = plt.figure(figsize=(8, 8))
plt.plot(x_data2,y_data,'ko',alpha=alpha_level)
plt.xlabel(r"x$_2$",fontsize=13)
plt.ylabel(r"y(x)",fontsize=13)
plt.title(r"$N_{samples}=$%i" %(NSAMPLE1),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);
"""


# =============================================================================
# A simple dense layer NN: predictions on data set 1

NMULTI = 2

# 1) place holders as variables that will eventually hold data, to do symbolic 
# computations on the graph later:
x = tf.placeholder(dtype=tf.float32, shape=[None,NMULTI]) # shape=(batch_size, 1)
y = tf.placeholder(dtype=tf.float32, shape=[None,1])

# 2) construct/initialize a neural network one-hidden layer and 20 units:
NHIDDEN = 24
W = tf.Variable(tf.random_normal([NMULTI,NHIDDEN], stddev=1.0, dtype=tf.float32))
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
NEPOCH = 6000
loss = np.zeros(NEPOCH) # store the training progress here
for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: X_DATA, y: y_data})
  loss[i] = sess.run(lossfunc, feed_dict={x: X_DATA, y: y_data})

plotname = figure_path +'convergence_3d_nn.png' 
plt.figure(figsize=(8, 8))
plt.plot(np.arange(100, NEPOCH,1), loss[100:], 'b-')
plt.ylabel(r"loss",fontsize=13)
plt.xlabel(r"$N_{epoch}$",fontsize=13)
plt.title(r"$N_{samples}=$%i, $N_{hidden}=$%i, $N_{epochs}=$%i" %(NSAMPLE1,NHIDDEN,NEPOCH),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

y_train = sess.run(y_out,feed_dict={x: X_DATA})

sess.close() # close the session afterwards to free resources 

plotname = figure_path +'training_prediction_3d_nn.png' 
fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
ax.scatter3D(x_data, x_data2, y_data, zdir='z', alpha=0.2, s=20, c='k') 
ax.scatter3D(x_data, x_data2, y_train, zdir='z', alpha=0.2, s=20, c='r') 
ax.set_xlabel(r"x$_1$",fontsize=13)
ax.set_ylabel(r"x$_2$",fontsize=13)
ax.set_zlabel(r"y",fontsize=13)
ax.view_init(elev=20., azim=130)
ax.set_xlim3d(-15, 15)
ax.set_ylim3d(-15, 15)
ax.set_zlim3d(-20,20)
plt.title(r"NN training result 1, $N_{samples}=$%i, $N_{hidden}=$%i, $N_{epochs}=$%i" %(NSAMPLE1,NHIDDEN,NEPOCH),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);


# =============================================================================
# make data set 2

temp_data = x_data
x_data = y_data
y_data = temp_data
X_DATA = np.hstack((x_data,x_data2))

plotname = figure_path +'training_data_3d_2.png' 
fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
zdata = y_data
ydata = x_data2
xdata = x_data
ax.scatter3D(xdata, ydata, zdata, zdir='z', alpha=0.2, s=20, c='k') 
ax.set_xlabel(r"x$_1$",fontsize=13)
ax.set_ylabel(r"x$_2$",fontsize=13)
ax.set_zlabel(r"y",fontsize=13)
ax.view_init(elev=5., azim=20)
ax.set_xlim3d(-15, 15)
ax.set_ylim3d(-15, 15)
ax.set_zlim3d(-20,20)
plt.title(r"training data 2, $N_{samples}=$%i" %(NSAMPLE1),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);


# =============================================================================
# A simple dense layer NN: predictions on data set 2

NMULTI = 2

# 1) place holders as variables that will eventually hold data, to do symbolic 
# computations on the graph later:
x = tf.placeholder(dtype=tf.float32, shape=[None,NMULTI]) # shape=(batch_size, 1)
y = tf.placeholder(dtype=tf.float32, shape=[None,1])

# 2) construct/initialize a neural network one-hidden layer and 20 units:
NHIDDEN = 24
W = tf.Variable(tf.random_normal([NMULTI,NHIDDEN], stddev=1.0, dtype=tf.float32))
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
NEPOCH = 6000
loss = np.zeros(NEPOCH) # store the training progress here
for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: X_DATA, y: y_data})
  loss[i] = sess.run(lossfunc, feed_dict={x: X_DATA, y: y_data})

plotname = figure_path +'convergence_3d_nn2.png' 
plt.figure(figsize=(8, 8))
plt.plot(np.arange(100, NEPOCH,1), loss[100:], 'b-')
plt.ylabel(r"loss",fontsize=13)
plt.xlabel(r"$N_{epoch}$",fontsize=13)
plt.title(r"$N_{samples}=$%i, $N_{hidden}=$%i, $N_{epochs}=$%i" %(NSAMPLE1,NHIDDEN,NEPOCH),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

y_train = sess.run(y_out,feed_dict={x: X_DATA})

sess.close() # close the session afterwards to free resources 

plotname = figure_path +'training_prediction_3d_nn2.png' 
fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
ax.scatter3D(x_data, x_data2, y_data, zdir='z', alpha=0.2, s=20, c='k') 
ax.scatter3D(x_data, x_data2, y_train, zdir='z', alpha=0.2, s=20, c='r') 
ax.set_xlabel(r"x$_1$",fontsize=13)
ax.set_ylabel(r"x$_2$",fontsize=13)
ax.set_zlabel(r"y",fontsize=13)
ax.view_init(elev=5., azim=20)
ax.set_xlim3d(-15, 15)
ax.set_ylim3d(-15, 15)
ax.set_zlim3d(-20,20)
plt.title(r"NN training result 2, $N_{samples}=$%i, $N_{hidden}=$%i, $N_{epochs}=$%i" %(NSAMPLE1,NHIDDEN,NEPOCH),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);



# =============================================================================
# A mixture density network


# construct the MDN:
NHIDDEN = 24
STDEV = 0.5
KMIX = 24 # number of mixtures
NOUT = KMIX * 3 # pi, mu, stdev

x = tf.placeholder(dtype=tf.float32, shape=[None,NMULTI], name="x") 
y = tf.placeholder(dtype=tf.float32, shape=[None,1], name="y")

Wh = tf.Variable(tf.random_normal([NMULTI,NHIDDEN], stddev=STDEV, dtype=tf.float32))
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

# = 6000 for NHIDDEN = 24, STDEV = 0.5, KMIX = 24
NEPOCH = 10000 
loss = np.zeros(NEPOCH) # store the training progress here.
for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: X_DATA, y: y_data})
  loss[i] = sess.run(lossfunc, feed_dict={x: X_DATA, y: y_data})

plotname = figure_path +'convergence_3d_mdn.png' 
plt.figure(figsize=(8, 8))
plt.plot(np.arange(100, NEPOCH,1), loss[100:], 'b-')
plt.ylabel(r"loss",fontsize=13)
plt.xlabel(r"$N_{epoch}$",fontsize=13)
plt.title(r"$N_{samples}=$%i, $N_{hidden}=$%i, $N_{epochs}=$%i, $N_{mix}=$%i" %(NSAMPLE1,NHIDDEN,NEPOCH,KMIX),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);

# testing:
#x_test = np.float32(np.arange(-15,15,0.1))
#NTEST = x_test.size
#x_test = x_test.reshape(NTEST,1) # needs to be a matrix, not a vector

out_pi_test, out_sigma_test, out_mu_test = sess.run(fn.get_mixture_coeff(output,KMIX), feed_dict={x: X_DATA})

y_test = fn.generate_ensemble( out_pi_test, out_mu_test, out_sigma_test, X_DATA , 1 )

sess.close()

plotname = figure_path +'training_prediction_3d_mdn.png' 
fig = plt.figure(figsize=(8, 8))
ax = Axes3D(fig)
ax.scatter3D(x_data, x_data2, y_data, zdir='z', alpha=0.2, s=20, c='k') 
ax.scatter3D(x_data, x_data2, y_test, zdir='z', alpha=0.2, s=20, c='r') 
ax.set_xlabel(r"x$_1$",fontsize=13)
ax.set_ylabel(r"x$_2$",fontsize=13)
ax.set_zlabel(r"y",fontsize=13)
ax.view_init(elev=5., azim=20)
ax.set_xlim3d(-15, 15)
ax.set_ylim3d(-15, 15)
ax.set_zlim3d(-20,20)
plt.title(r"MDN training result 2, $N_{samples}=$%i, $N_{hidden}=$%i, $N_{epochs}=$%i" %(NSAMPLE1,NHIDDEN,NEPOCH),fontsize=13)
plt.savefig(plotname,format="png"); plt.close(fig);










