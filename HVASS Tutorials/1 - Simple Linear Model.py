import matplotlib.pyplot as plt
import tensorflow as tf
device_name = "/cpu:0"
import numpy as np
from sklearn.metrics import confusion_matrix

# Importing MNIST Data set images
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets("data/MNIST", one_hot=True)

# Convert the one-hot array into single numbers based on the index of the highest element
data.test.cls = np.array([label.argmax() for label in data.test.labels])

# MNIST images are 28 pixels
IMG_SIZE = 28

# They are stored in one-dimensional arrays based on the image size
IMG_SIZE_FLAT = IMG_SIZE * IMG_SIZE

# Using a tuple with height and width, makes a variable to reshape arrays
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)

# NUM Classes, one for each of the 10 digits possible
NUM_CLASSES = 10

# Number of Iterations for training
NUM_EPOCHS = 1000


# Helper class to plot 9 images with its predicted and true classes
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(IMG_SHAPE), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


def test_plot():
    images = data.test.images[0:9]
    cls_true = data.test.cls[0:9]
    plot_images(images, cls_true)

######################################

# Tensor vector that holds an arbitrary number of images with each image being a vector of length img_size_flat
x = tf.placeholder(tf.float32, [None, IMG_SIZE_FLAT])

# Tensor vector that holds the True labels for the images
labels = tf.placeholder(tf.float32, [None, NUM_CLASSES])
labels_cls = tf.placeholder(tf.int64, [None])

# Variables to be optimized
weights = tf.Variable(tf.zeros([IMG_SIZE_FLAT, NUM_CLASSES]))

biases = tf.Variable(tf.zeros(NUM_CLASSES))

# Multiplies the images in x with the weights then adds the biases, to find activation of the tensors in the next layer
logits = tf.matmul(x, weights) + biases

# Since the numbers resulting from the logits calculation are extremely high or low, we normalize the values between
# one and zero using a softmax function, like a sigmoid function
labels_prediction = tf.nn.softmax(logits=logits)
labels_prediction_cls = tf.argmax(labels_prediction, axis=1)

# Cost function to optimize, the indicator of how far off the model is off
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
cost = tf.reduce_mean(cross_entropy)

# Optimizing function to adjust the weights and biases in order to bring the cost down and improve accuracy
# Utilizes a Gradient Descent algorithm to try and reach the local minimum
# Learning rate adjusts how far a step down the curve it goes
optimizer = tf.train.AdamOptimizer(learning_rate=.5).minimize(cost)

# Performance measurements
# Returns a boolean whether the firing neuron of the neural net is equal to the true value
correct_prediction = tf.equal(labels_prediction_cls, labels_cls)

# Calculates the accuracy of the prediction
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#############################
# Session RunTime
#############################
with tf.device(device_name):
    with tf.Session() as sess:
        # Initialize all the variables
        sess.run(tf.global_variables_initializer())

        batch_size = 1000

        for i in range(NUM_EPOCHS):
            x_batch, labels_batch = data.train.next_batch(batch_size)

            feed_dict_train = {x: data.test.images,
                               labels: data.test.labels}

            sess.run(optimizer, feed_dict=feed_dict_train)

        feed_dict_test = {x: data.test.images,
                          labels: data.test.labels,
                          labels_cls: data.test.cls}

        # Calculating and Printing accuracy
        acc = sess.run(accuracy, feed_dict=feed_dict_test)
        print("Accuracy on test-set: ", acc)

        sess.close()


##########################
# Results from Different Experiments
#
# Using GradientDescent:
# Learning Rate: .50, Epochs: 1000, Acc: 94.0%
# Learning Rate: .75, Epochs: 1000, Acc: 94.5%
# Learning Rate: 1.0, Epochs: 1000, Acc: 94.8%
#
# Using AdamOptimizer:
# Learning Rate: .5, Epochs: 1000, Acc: 100%
#
# Using AdagradOptimizer:
# Learning Rate: .5, Epochs: 1000, Acc: 95.4%
##########################
