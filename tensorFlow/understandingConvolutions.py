# Convolution: 1D operation with Python (Numpy/Scipy)
import numpy as np
h = [2,1,0] # ex.image
x = [3,4,5] # kernel
y = np.convolve(x,h) # array([ 6, 11, 14,  5,  0])
print("Compare with the following values from Python: y[0] = {0} ; y[1] = {1}; y[2] = {2}; y[3] = {3}; y[4] = {4}".format(y[0],y[1],y[2],y[3],y[4])) 

# 1) Visually understanding the operation with padding (full)
import numpy as np
x= [6,2] #  invert the filter x, otherwise the operation would be cross-correlation.
h= [1,2,5,4] # 0 [1  2  5  4] 0 
y= np.convolve(x,h,"full")  #now, because of the zero padding, the final dimension of the array is bigger
y  # array([ 6, 14, 34, 34, 8])

# 2) Visually understanding the operation with "same"
import numpy as np
x= [6,2]
h= [1,2,5,4] # just add the zero to left > 0 [1  2  5  4]
y= np.convolve(x,h,"same")  #it is same as zero padding, but withgenerates same 
y # array([ 6, 14, 34, 34])

# 3) Visually understanding the operation with no padding (valid)
import numpy as np
x= [6,2]
h= [1,2,5,4]
y= np.convolve(x,h,"valid")  #we will understand why we used the argument valid in the next example
y # array([14, 34, 34])

# Convolution: 2D operation with Python (Numpy/Scipy)
from scipy import signal as sg
I= [[255,   7,  3],
    [212, 240,  4],
    [218, 216, 230],]
g= [[-1,1]]
print ('Without zero padding \n')
print ('{0} \n'.format(sg.convolve( I, g, 'valid')))
# The 'valid' argument states that the output consists only of those elements 
# that do not rely on the zero-padding.
print ('With zero padding \n')
print (sg.convolve(I, g))

from scipy import signal as sg
I= [[255,   7,  3],
    [212, 240,  4],
    [218, 216, 230],]
g= [[-1,  1],
    [ 2,  3],]
print ('With zero padding \n')
print ('{0} \n'.format(sg.convolve( I, g, 'full')))
# The output is the full discrete linear convolution of the inputs. 
# It will use zero to complete the input matrix

print ('With zero padding_same_ \n')
print ('{0} \n'.format(sg.convolve( I, g, 'same')))
# The output is the full discrete linear convolution of the inputs. 
# It will use zero to complete the input matrix

print ('Without zero padding \n')
print (sg.convolve( I, g, 'valid'))
# The 'valid' argument states that the output consists only of those elements 
#that do not rely on the zero-padding.


# Coding with TensorFlow
import tensorflow as tf

#Building graph
input = tf.Variable(tf.random_normal([1,10,10,1]))
filter = tf.Variable(tf.random_normal([3,3,1,1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

#Initialization and session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print("Input \n")
    print('{0} \n'.format(input.eval()))
    print("Filter/Kernel \n")
    print('{0} \n'.format(filter.eval()))
    print("Result/Feature Map with valid positions \n")
    result = sess.run(op)
    print(result)
    print('\n')
    print("Result/Feature Map with padding \n")
    result2 = sess.run(op2)
    print(result2)

# Convolution applied on images
# download standard image
!wget --quiet https://ibm.box.com/shared/static/cn7yt7z10j8rx6um1v9seagpgmzzxnlz.jpg --output-document bird.jpg

#Importing
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

### Load image of your choice on the notebook
print("Please type the name of your test image after uploading to \
your notebook (just drag and grop for upload. Please remember to \
type the extension of the file. Default: bird.jpg")

raw= raw_input()
im = Image.open(raw)  # type here your image's name

# uses the ITU-R 601-2 Luma transform (there are several 
# ways to convert an image to grey scale)

image_gr = im.convert("L")    
print("\n Original type: %r \n\n" % image_gr)

# convert image to a matrix with values from 0 to 255 (uint8) 
arr = np.asarray(image_gr) 
print("After conversion to numerical representation: \n\n %r" % arr) 
### Activating matplotlib for Ipython
%matplotlib inline

### Plot image
imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')  #you can experiment different colormaps (Greys,winter,autumn)
print("\n Input image converted to gray scale: \n")
plt.show(imgplot)

kernel = np.array([
                        [ 0, 1, 0],
                        [ 1,-4, 1],
                        [ 0, 1, 0],
                                     ]) 

grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')

%matplotlib inline
print('GRADIENT MAGNITUDE - Feature map')
fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')

type(grad)
grad_biases = np.absolute(grad) + 100
grad_biases[grad_biases > 255] = 255

%matplotlib inline
print('GRADIENT MAGNITUDE - Feature map')
fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad_biases), cmap='gray')