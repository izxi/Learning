# Exercise-Linear Regression with TensorFlow
#   This data is derieved from the above mentioned book and aim is to fit a linear model and predict the "Best Fit Line" 
#   for the given "Chirps(per 15 Second)" in Column 'A' and the corresponding "Temperatures(Farenhite)" in Column 'B' using TensorFlow.
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#downloading dataset
!wget -nv -O /resources/data/PierceCricketData.csv https://ibm.box.com/shared/static/reyjo1hk43m2x79nreywwfwcdd5yi8zu.csv
df = pd.read_csv("/resources/data/PierceCricketData.csv")
df.head()

%matplotlib inline
x_data, y_data = (df["Chirps"].values,df["Temp"].values)
plt.plot(x_data, y_data, 'ro') # plots the data points
plt.xlabel("# Chirps per 15 sec") # label the axis
plt.ylabel("Temp in Farenhiet")
# Create a Data Flow Graph using TensorFlow
X = tf.placeholder(tf.float32, shape=(x_data.size))
Y = tf.placeholder(tf.float32,shape=(y_data.size))
m = tf.Variable(3.0) # tf.Variable call creates a single updatable copy in the memory and efficiently updates 
c = tf.Variable(2.0) # the copy to relfect any changes in the variable values through out the scope of the tensorflow session
Ypred = tf.add(tf.multiply(X, m), c) # Construct a Model
# Create and Run a Session to Visualize the Predicted Line from above Graph
session = tf.Session() #create session and initialize variables
session.run(tf.global_variables_initializer())
pred = session.run(Ypred, feed_dict={X:x_data}) #get prediction with initial parameter values
plt.plot(x_data, pred) #plot initial prediction against datapoints
plt.plot(x_data, y_data, 'ro')
plt.xlabel("# Chirps per 15 sec") # label the axis
plt.ylabel("Temp in Farenhiet")

# Define a Graph for Loss Function
nf = 1e-1 # normalization factor
loss = tf.reduce_mean(tf.squared_difference(Ypred*nf,Y*nf)) # seting up the loss function

# Define an Optimization Graph to Minimize the Loss and Training the Model
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#optimizer = tf.train.AdagradOptimizer(0.01 )
train = optimizer.minimize(loss) # pass the loss function that optimizer should optimize on.
session.run(tf.global_variables_initializer()) # Initialize all the vairiables again

# Run session to train and predict the values of 'm' and 'c' for different training steps along with storing the losses in each step
convergenceTolerance = 0.0001
previous_m = np.inf
previous_c = np.inf
steps = {}
steps['m'] = []
steps['c'] = []
losses=[]
for k in range(100000):
    _, _m , _c,_l = session.run([train, m, c,loss],feed_dict={X:x_data,Y:y_data}) # run a session to train , get m and c values with loss function 
    steps['m'].append(_m)
    steps['c'].append(_c)
    losses.append(_l)
    if (np.abs(previous_m - _m) <= convergenceTolerance) or (np.abs(previous_c - _c) <= convergenceTolerance):
        print "Finished by Convergence Criterion"
        print k
        print _l
        break
    previous_m = _m, 
    previous_c = _c, 
session.close() 

plt.plot(losses[:]) # Print the loss function
