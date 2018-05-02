# Activation Functions
#   Activation Functions define how a processing unit will treat its input -- 
#   usually passing this input through it and generating an output through its result. 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
# implements a basic function that plots a surface for an arbitrary activation function. 
def plot_act(i=1.0, actfunc=lambda x: x):
    ws = np.arange(-0.5, 0.5, 0.05) # between -0.5 and 0.5 with a step of 0.05
    bs = np.arange(-0.5, 0.5, 0.05)

    X, Y = np.meshgrid(ws, bs)

    os = np.array([actfunc(tf.constant(w*i + b)).eval(session=sess) \
                   for w,b in zip(np.ravel(X), np.ravel(Y))])

    Z = os.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1)
# Basic Structure
#start a session
sess = tf.Session();
#create a simple input of 3 real values
i = tf.constant([1.0, 2.0, 3.0], shape=[1, 3])
#create a matrix of weights
w = tf.random_normal(shape=[3, 3])
#create a vector of biases
b = tf.random_normal(shape=[1, 3])
#dummy activation function
def func(x): return x
#tf.matmul will multiply the input(i) tensor and the weight(w) tensor then sum the result with the bias(b) tensor.
act = func(tf.matmul(i, w) + b)
#Evaluate the tensor to a numpy array
act.eval(session=sess) # array([[-4.7375083,  1.1410106, -2.2912326]], dtype=float32)

plot_act(1.0, func)

# The Step Functions
#   The Step function was the first one designed for Machine Learning algorithms.
#   It consists of a simple threshold function that varies the Y value from 0 to 1.
#   Tensorflow dosen't have a Step Function.
# The Sigmoid Functions
# Sigmoid functions are very useful in the sense that they "squash" their given inputs into a bounded interval.
# This is exceptionally handy when combining these functions with others such as the Step function.
plot_act(1, tf.sigmoid)
# Using sigmoid in a neural net layer
act = tf.sigmoid(tf.matmul(i, w) + b)
act.eval(session=sess)
# TanH is widely used in a wide range of applications,
# and is probably the most used function of the Sigmoid family.
plot_act(1, tf.tanh)
# Using tanh in a neural net layer
act = tf.tanh(tf.matmul(i, w) + b)
act.eval(session=sess) # array([[-0.9998984 ,  0.24348633, -0.9873636 ]], dtype=float32)

# The Linear Unit functions
# Linear Units in general tend to be variation of what is called the Rectified Linear Unit, or ReLU for short.
# The ReLU is a simple function which operates within the  [0,∞)  interval.
#   For the entirety of the negative value domain, it returns a value of 0, 
#   while on the positive value domain, it returns  x  for any  f(x) .
#      the ReLU structure takes care of what is called the Vanishing and Exploding Gradient problem by itself.
plot_act(1, tf.nn.relu)
# Using relu in a neural net layer
act = tf.nn.relu(tf.matmul(i, w) + b)
act.eval(session=sess) # array([[0., 0., 0.]], dtype=float32)
