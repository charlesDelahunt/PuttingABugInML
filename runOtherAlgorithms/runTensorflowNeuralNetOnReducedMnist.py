#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runTensorflowNeuralNetOnReducedMnist.py:
Trains a neural net on a reduced (downsampled) MNIST dataset. The reduction of the MNIST thumbnails accords with 
the reduction used in moth training experiments at github/puttingABugInML.

This code is a mashup of: one of Tensorflow's mnist examples; a script by Christian Szegedy; and my own code
Any poor syntax or inferior results are strictly my own responsibility.

Charles B. Delahunt. delahunt@uw.edu
"""

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import matplotlib as mpl
from skimage.transform import resize

#---------------------------------------------------------------------
# KEY USER ENTRIES: 
trainPerClass = 1 # training samples per class

numImsToView = 30  # to view post-processed thumbnails. 0 means do not view
printNetworkInfo = True
numGlimpses = 5  # how many training updates. MUST BE > 0
#---------------------------------------------------------------------

# Misc parameters:
numClasses = 10
testPerClass = 15  # to match numbers in moth experiments   
numTestImages = testPerClass*numClasses 
testBatchSize = numTestImages 
numTrainImages = trainPerClass*numClasses

# Pre-processing parameters:
numMeanSubtractImagesPerClass = 500
pixelSum = 6  # normalize so that each images' pixels sum to this
numFeatures = 85 # % of pixels (in downsampled images) to dropoutKeepFraction

reduceImagesFlag =  True # False #
crop = 2   # number of pixels to remove from each side
downsampleRate = 2    # downsample rate
side = 28    # since raw mnist thumbnails are 28 x 28

# Parameters for the neural net:
noise = 0.01
dropoutKeepFraction = 0.7  # equivalent to DROPOUT = 0.3
#learningRate = [0.8, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001] # for sweep
learningRate = 0.01   # learning rate
numEpochs = 50 # 100 
# number of units in the hidden layer will equal the number of active pixels.

#--------------------------------------------------------------------------------

#%% Some functions:

''' 
preprocess_fn(imStack, crop, downsampleRate, pixelSum, globalMeanIm, side: 28):
    For each image in stack:
    Crop, downsample, normalize, mean-subtract, make non-negative, and make pixels sum to a constant.
    This fn does not include the retaining of just active pixel indices.
    Not sure what kind of downsampling skimage.transform.resize does.
'''
def preProcess_fn ( imStack, crop, downsampleRate, pixelSum, globalMeanIm, side: 28 ):
        
        # first prepare globalMeanIm: a bit inefficient to do it here, but maintains argin unity
        globalMeanIm.shape = ( side, side)
        globalMeanIm = globalMeanIm[crop: -crop, crop:-crop ]
        
        newSide = int( globalMeanIm.shape[0] / downsampleRate )
        
        globalMeanIm = resize(globalMeanIm, (   newSide, newSide )  ) # downsample
        
        # now process imStack:
        imStack.shape = ( imStack.shape[0], side, side)    # reshape    28**2 = 784
    
        imStack = imStack[ :,crop: -crop, crop:-crop ]  # crop
        
        newImStack = np.zeros([ imStack.shape[0], int(newSide*newSide) ])  # initialize
        
        for i in range(0, imStack.shape[0]):
            im = imStack[i,:,:]         # 2-d array
            newIm = resize(im, (newSide, newSide ) )  # downsample
            newIm = newIm.flat / max(newIm.flat)    # now a vector
            newIm = newIm - globalMeanIm.flat   
            newIm = np.maximum( newIm, 0 ) 
            newIm = newIm * pixelSum / sum(newIm.flat)
            newImStack[i,:] = newIm 
            
        return newImStack
    
#------------------------------------------------------------------------------------  
'''
selectActivePixels_fn chooses numFeatures pixels that have highest values in imStack
Typical use: argin 1 = classMeanIms
'''
def selectActivePixels_fn(imStack, numFeatures ):     
    # Method 1: determine the active pixels for each class average:
    # 1. select a threshold, and dropoutKeepFraction all pixels that are higher than threshold in any classAve.
    # 2. combine the active pixels from the class aves (there will be overlap)
    # 3. lower threshold, repeat.
    # 4. when we have 'numFeatures' active pixels, stop.
    
    # sort all the pixel values descending order:
    pixelVals = np.sort(imStack.flat)
    pixelVals = np.unique(pixelVals)
    pixelVals = np.flipud(pixelVals)
    
    activePixelInds = list( [] )
    
    # use the highest pixel value as a threshold, and select all pixels with values >= thresh. Repeat 
    while True:
        
        thresh = pixelVals[0]
        pixelVals = pixelVals[1:-1]  # discard this value 
        
        for i in range(10):
            thisIm = imStack[i, :]
            for j in range( len(thisIm) ):
                if thisIm[j] >= thresh:
                    activePixelInds.append(j)  # all-class list of active inds
                    thisIm[j] = 0             # void the image entry
                    
        activePixelInds = list( set( activePixelInds) )  # remove duplicates
        if  ( len(activePixelInds) > numFeatures ):
            break
        
        # end while            
                    
    return activePixelInds

#---------------------------------------------------------------------   
''' 
viewThumbnails_fn(imStack, numImsToView, numClasses, side )  
    # View some thumbnails from test set to see how they look
'''
def viewThumbnails_fn(imStack, numImsToView, numClasses, activePixelInds, side):
    if (numImsToView > 0):
        viewInds = np.random.choice(imStack.shape[0], size=numImsToView,replace=False)
        numPerRow = 15
        rows = int(numImsToView / numPerRow )  # ignore surplus
        for i in range(rows):
            viewIms = np.zeros( [ numPerRow, newSide*newSide ] )
            rowViewInds = viewInds[ i*numClasses : i*numClasses + numPerRow ] 
            viewIms [:, activePixelInds ] = imStack [  rowViewInds, :  ]  # substitute in the active pixel values
            viewIms.shape = (numPerRow, newSide, newSide)
            rowOfIms = np.zeros( [ newSide, numPerRow*newSide ] )
            for i in range(viewIms.shape[0]):
                rowOfIms[ :, i*newSide:(i + 1)*newSide ] = viewIms[i,:,:] / max( (viewIms[i,:,:]).flat ) 
                # normalize thumbs to [0, 1] for viewing
        
            mpl.pyplot.gray()
            mpl.pyplot.figure(figsize=(16, 12))
            mpl.pyplot.imshow(rowOfIms)

'''
################################################################################ 

#%% MAIN:

'''            
# clear previous graph:
tf.reset_default_graph()

sess = tf.InteractiveSession()

# load full data
xTrain, yTrain, xVal, yVal, xTest, yTest = tl.files.load_mnist_dataset(shape=(-1,784))

''' 
Divide the images into piles (train, val, test, meanSubtract, classAves):
'''

train_indices_by_class = list([] for i in range(10))
for (i, l) in enumerate(yTrain):
   train_indices_by_class[l].append(i)
        
# training sample indices and mean-subtraction indices, equal numbers per class:
train_indices_by_class = [np.random.choice(l, trainPerClass + numMeanSubtractImagesPerClass, replace=False) for l in train_indices_by_class]
mean_subtract_indices_by_class =  list([] for i in range(10))
# divide these into train and mean-subtraction groups:
for i in range(len(train_indices_by_class)):
    t = train_indices_by_class[i]
    t = t[ trainPerClass - 1:-1]
    mean_subtract_indices_by_class[i] = t
    
    t = train_indices_by_class[i]
    t = t[ 0:trainPerClass ]
    train_indices_by_class[i] = t

train_indices = [index for l in train_indices_by_class for index in l]
mean_subtract_indices = [index for l in mean_subtract_indices_by_class for index in l]
   
# test sample indices, equal numbers per class: This makes test_indices a list.
test_indices_by_class = list([] for i in range(10)) 
for (i, l) in enumerate(yTest):
   test_indices_by_class[l].append(i)
test_indices_by_class = [np.random.choice(l, size=testPerClass, replace=False) for l in test_indices_by_class]
test_indices = [index for l in test_indices_by_class for index in l]
   
# val (used for reports during nn training) sample indices, equal numbers per class: Use same number as train samples to avoid crash
val_indices_by_class = list([] for i in range(10)) 
for (i, l) in enumerate(yVal):
   val_indices_by_class[l].append(i)
val_indices_by_class = [np.random.choice(l, size=trainPerClass, replace=False) for l in val_indices_by_class]
val_indices = [index for l in val_indices_by_class for index in l] 

# using the above indices, make X_train and X_test smaller.
meanSubtractIms = xTrain[mean_subtract_indices,:]  
meanSubtractLabels = yTrain[mean_subtract_indices,]
xTrain = xTrain[train_indices, :]
yTrain = yTrain[train_indices,  ]
xTest = xTest[test_indices, ]
yTest = yTest[test_indices, ]
xVal = xVal[val_indices, ]     # X_val does not matter, but let's shrink it anyway.
yVal = yVal[val_indices, ] 

# redefine mean_subtract_indices to refer to the new, smaller set:
mean_subtract_indices_by_class = list([] for i in range(10)) 
for (i, l) in enumerate(meanSubtractLabels):
   mean_subtract_indices_by_class[l].append(i)
mean_subtract_indices = [index for l in mean_subtract_indices_by_class for index in l] 

newSide = int( ( np.sqrt( xTrain.shape[1] ) - 2*crop ) / downsampleRate )     # for future ref if needed. Should be an int.

''' 
########################################################
#%% We now have image stacks and label vectors for train, test, meanSubtraction images, and val.
Preprocess images if indicated:
'''

# First define a needed variable. It will be replaced during preprocessing if there is any:
activePixelInds = range(784)
     
if reduceImagesFlag: 
# Case: Preprocess the images to reduce the number of pixels
    
    # first create a global mean image, so we can mean-subtract on other images:
    globalMeanIm = np.zeros( [ meanSubtractIms.shape[1]  ] )
    for i in range(meanSubtractIms.shape[0]):
        globalMeanIm = globalMeanIm + meanSubtractIms[i,:]
        
    globalMeanIm = globalMeanIm / meanSubtractIms.shape[0]
    
    # Now preprocess various stacks:
    xTrain = preProcess_fn(xTrain, crop, downsampleRate, pixelSum, globalMeanIm, side)    
    xVal = preProcess_fn(xVal, crop, downsampleRate, pixelSum, globalMeanIm, side)
    xTest = preProcess_fn(xTest, crop, downsampleRate, pixelSum, globalMeanIm, side)
    
    # create and preprocess the 10 class average images, for use in selecting active pixels:
    # first create a global mean image, so we can mean-subtract on other images:
    classMeanIms = np.zeros( [ numClasses, meanSubtractIms.shape[1] ])  # initialize
    for i in range(numClasses):
        classMeanIms[ i, : ] = np.sum( meanSubtractIms[ mean_subtract_indices_by_class[i], : ], 0)        
        
    classMeanIms = preProcess_fn( classMeanIms, crop, downsampleRate, pixelSum, globalMeanIm, side )
    
    # Select the active pixels:
    activePixelInds = selectActivePixels_fn(classMeanIms, numFeatures ) 
    print(len(activePixelInds))    # sanity check 
 
    # reduce the various sets:
    xTrain = xTrain[ :, activePixelInds  ]
    xVal = xVal[ :, activePixelInds  ]    
    xTest = xTest[ :, activePixelInds  ]

# end if reduceImagesFlag
    
# view some of the images to be used
if (numImsToView > 0):
    viewThumbnails_fn(xTest, numImsToView, numClasses, activePixelInds, newSide) 

''' 
#####################################################################
#%% Define the neural network:
'''
    
# add noise to training images:
xTrain = xTrain + noise * np.random.normal(0, 1.0, xTrain.shape) # mean = 0, std = 1 
                                
# define placeholder
x = tf.placeholder(tf.float32, shape=[None, len(activePixelInds)], name='x')
y_ = tf.placeholder(tf.int64, shape=[None], name='y_')

# define the network
network = tl.layers.InputLayer(x, name='input')
network = tl.layers.DropoutLayer(network, keep=dropoutKeepFraction, name='drop1')
network = tl.layers.DenseLayer(network, len(activePixelInds), tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=dropoutKeepFraction, name='drop2')
#network = tl.layers.DenseLayer(network, len(activePixelInds), tf.nn.relu, name='relu2') # extra layer if wished
#network = tl.layers.DropoutLayer(network, keep=dropoutKeepFraction, name='drop3') 

# the softmax is implemented internally in tl.cost.cross_entropy(y, y_) to speed up computation, so we use identity here.
# see tf.nn.sparse_softmax_cross_entropy_with_logits()
network = tl.layers.DenseLayer(network, n_units=10, act=tf.identity, name='output')

# define cost function and metric.
y = network.outputs
cost = tl.cost.cross_entropy(y, y_, name='cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# define the optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost, var_list=train_params)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

# print network information
if (printNetworkInfo):
	network.print_params()
	network.print_layers()

#%% Train the network
tl.utils.fit(
    sess, network, train_op, cost, xTrain, yTrain, x, y_, acc=acc, batch_size=numTrainImages, n_epoch=numEpochs, print_freq=numEpochs/numGlimpses, X_val=xVal, y_val=yVal, eval_train=False)

# evaluation
tl.utils.test(sess, network, acc, xTest, yTest, x, y_, batch_size=None, cost=cost)

# save the network to .npz file
tl.files.save_npz(network.all_params, name='model.npz')
sess.close()
