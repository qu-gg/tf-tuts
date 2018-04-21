import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from tensorflow.examples.tutorials.mnist import input_data


# Convolutional Layer 1
filter_size1 = 5            # Convolution filters are 5x5 pixels
num_filters1 = 16           # There are 16 of these filters

# Layer 2
filter_size2 = 5
num_filters2 = 36           # There are 36 of these filters in the second layer

# Fully Connected Layer
fc_size = 128               # Number of neurons in the fully-connected layer

# MNIST Data set
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

print("Size of ")
print(" - Training Set: ", len(data.train.labels))
print(" - Test Set: ", len(data.test.labels))
print(" - Validation Set:", len(data.validation.labels))

# Class labels, 0-9 one hot encoded.
data.test.cls = np.argmax(data.test.labels, axis = 1)

# MNIST properties
img_size = 28                           # Size of one image

img_size_flat = img_size * img_size     # Images are stored in 1d array

img_shape = (img_size, img_size)        # Tuple of height and weight, used to reshape arrays

num_channels = 1                        # Num of color channels for images: 1 channel for gray-scale

num_classes = 10                        # Num of classes, one for each 10 digits

# Functions for creating new TensorFlow variables in a given shape and with random values
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

"""
Helper function to create a new convolutional layer
"""
def new_conv_layer(prev_layer,              # Previous layer
                   num_input_channels,      # Num. channels in prev. layer
                   filter_size,             # Width and height of each filter
                   num_filters,             # Num filters
                   use_pooling=True):


    # Shape of the filter-weights for the convolution
    # This format is determined by the TensorFlow API
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights, aka filters with given shape
    weights = new_weights(shape)

    # Create new biases, one for each filter
    biases = new_biases(num_filters)

    # Create the TensorFlow operation for convolution
    # Note the strides are set to 1 in all dimensions
    # First and last stride must always be one
    # because the first is for image-number and the last
    # is for input channel (gray-scale)
    # But e.g. strides=[1,1,1,1] means that the filter is moved 2 pixels
    # across the x- and y-axis of the image
    # padding means the image is padded with zeros so the output size is the same
    layer = tf.nn.conv2d(input=prev_layer, filter=weights, strides=[1,1,1,1], padding='SAME')

    # Add the biases to the results of the convolution
    # A bias-value is added to each filter-channel
    layer += biases

    if use_pooling:
        # This is the 2x2 max-pooling
        layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # Rectified Linear Unit (ReLU)
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows
    # more complicated functions
    layer = tf.nn.relu(layer)

    # Usually ReLU is done before pooling, but since the end result is the same
    # we can save 75% of the relu-operations by max-pooling first

    # We return both the resulting layer and the filter-weihts
    # because we will plot the weights later
    return layer, weights


"""
Since the conv layer produces an output tensor of 4d, it needs to be shrunk to 2d for the
fully connected layer to be able to use it
"""
def flatten(layer):
    layer_shape = layer.get_shape()    # Get the layer's tensor shape

    # Shape of the layer is assumed to be:
    # layer_shape = [num_images, img_height, img_width, num_channels]

    # The number of features is: image height * width * num_channels (1)
    # TF has a function to calculate this
    num_features = layer_shape[1:4].num_elements()

    # Reshaping, the total size of the tensor remains the same
    layer_flat = tf.reshape(layer, [-1, num_features])

    # Shape of the flattened layer is:
    # [num_images, img height * width * num_channels]

    # Returns both the flattened layer and number of features
    return layer_flat, num_features


"""
Helper function for creating a new fully-connected layer in the computational graph for TF
Nothing is actually calculated yet, it's just adding to the graph

Input is assumed to be a 2-dim tensor of shape [num_images, num_inputs]
"""
def new_fc_layer(prev_layer, num_inputs, num_outputs, use_relu=True):
    weights = new_weights([num_inputs, num_outputs])
    biases = new_biases(num_outputs)

    # Calculate the layer as a matrix mul of the input and weights
    # then add the bias-values
    # This is calculating the strength of a single neuron in the layer
    # (neuron * weight) + bias = input to the next layer
    layer = tf.matmul(prev_layer, weights) + biases

    # Use ReLU?:
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# Placeholder variable for the input images, allows changing the images that are input
# to the TF graph. This is a 'tensor', which just means it is a multi-dim vector/matrix
# Shape is set to [None, img_size_flat], which means it can hold an arbitrary number of images
# and each image can have a vector of length img_size_flat
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

# The conv layer expects x to be a 4-dim tensor, so it has to be reshaped
# It's shape is [num_images, img_height, img_width, num_channels]
# img_height == img_width == img_size
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

# Next are the placeholder variables for the true labels associated with the images
# input to x
# The shape is [None, num_classes], which means it can hold any num of images and has 10 classes
y_true = tf.placeholder(tf.float32, shape=[None, num_classes])

# Instead of a placeholder for the class-number, it'll be calculated using argmax
y_true_cls = tf.argmax(y_true,axis=1)


"""
Conv Layer 1

It takes x_image as the input and creates num_filters1 different filters,
each having width and height equals for filter_size1
It is downsized at the end

Shape of this output will be [?, 14, 14, 16]
? = Arbitrary num of images
14 = pixels wide and high
16 = number of channels, one for each filter
"""
layer_conv1, weights_conv1 = \
    new_conv_layer(prev_layer=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

print(layer_conv1)

"""
Conv Layer 2

Takes input as the output from the first layer
Number of input channels corresponds to the number of filters in the first layer

Shape of this output will be [?, 7, 7, 36]
? = Arbitrary num of images
7 = pixels wide and high
36 = channels, one for each filter
"""
layer_conv2, weights_conv2 = \
    new_conv_layer(prev_layer=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

print(layer_conv2)

# Want to flatten the output of the second conv layer in order to put it through the
# fully-connected layer
layer_flat, num_features = flatten(layer_conv2)


"""
Fully-Connected Layer 1

Input is the flattened conv2 layer of 2 dimensions
Number of neurons in the layer is fc_size
ReLU is used so it can learn non-linear relations

Shape of this output is [?, 128]
? = Arbitrary num of images
128 = number of neurons in this layer
"""
layer_fc1 = new_fc_layer(prev_layer=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

"""
Fully-Connected Layer 2

Input is the fc layer 1
Number of neurons is 10, one for each of the potential classes
ReLU does not need to be used in this layer as it is the final output

Shape of this output is [?, 10]
? = Arbitrary num of images
10 = number of neurons in this layer
"""
layer_fc2 = new_fc_layer(prev_layer=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)


"""
Predicted Class
"""
# This smushes all the output values in the second fc layer into a number between
# 0 - 1, since these numbers can range from -infinity to positive infinity
y_pred = tf.nn.softmax(layer_fc2)

# The predicted class is the index of the largest element
y_pred_cls = tf.argmax(y_pred, axis=1)


"""
Cost Function

This is a performance measure, zero meaning 100% accuracy and anything above
that is inaccuracy.

Since the function interally calculates the softmax,the output from the second
fc layer needs to be used rather than what the y_pred is.
"""
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)

# Since the cross-entropy is calculated for each image, each inaccuracy is individual
# So, the average is taken of all the inaccuracies in order to use it in the optimization method
cost = tf.reduce_mean(cross_entropy)

"""
Optimization Method

Since the cost measure is found, it can be minimized using an optimization function
Choices include:
   - AdamOptimizer
   - GradientDescent
Different optimizing strategies produces different results for different data problems
"""
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

"""
Performance Measures for user

The optimizer and accuracy are calculated for the machine's use, but it is incomprehensible
for human/user use
"""
correct_predictions = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

"""
TF Run
"""
sess = tf.Session()

sess.run(tf.global_variables_initializer()) # Initialize the variables before optimizing them
train_batch_size = 64                       # Num of images for a single optimizing run
total_iterations = 0                        # Counter for total num  performed

def optimize(num_iterations):
    global total_iterations                 # Update the global variables

    start_time = time.time()                # Start-time used for printing time usage

    for i in range(total_iterations, total_iterations + num_iterations):
        # grab a batch of training examples
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the tf graph
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data
        # TF assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer
        sess.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations
        if i % 100 == 0:
            #Calc the accuracy on the training-set
            acc = sess.run(accuracy, feed_dict_train)

            msg = "Optimization iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            print(msg.format(i + 1, acc))

    total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


test_batch_size = 256
def print_test_accuracy():
    # Number of images in the test-set
    num_test = len(data.test.images)

    # Allocate an array for the predicted classes which
    # Will be calculated in batches and filled into this array
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

    # Starting index for the next batch is denoted i
    i = 0

    while i < num_test:
        # Ending index for the next batch is denoted j
        j = min(i + test_batch_size, num_test)

        # Get the images for the batch between i and j
        images = data.test.images[i:j, :]

        # Get their labels
        labels = data.test.labels[i:j, :]

        # Create a feed_dict
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class
        cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    cls_true = data.test.cls

    # Create a boolean array whether each image was correctly classified
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images
    # When summing a boolean array, false means 0 and true means 1
    correct_sum = correct.sum()

    # Classification accyract is the number of correctly classified
    # versus the total size
    acc = float(correct_sum) / num_test

    # Print it
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

print_test_accuracy()
optimize(10000)
print_test_accuracy()
