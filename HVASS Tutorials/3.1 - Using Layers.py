import tensorflow as tf
import numpy as np
import math
from skimage import color
from skimage import io

device = "/gpu:0"

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

x = tf.placeholder(tf.float32, [None, img_size_flat], 'x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

y_true = tf.placeholder(tf.float32, [None, num_classes], 'y_true')
y_true_cls = tf.argmax(y_true, axis=1)

"""
Layers API Convolutional Net Implementation
"""
net = x_image       # refers to the previous layer

""" FIRST CONV LAYER"""
# Input: input image
# Filter: 16 filters with size
# Kernal: 5x5
# Padding: maintains same image size by filling with zeros
# ReLU: True
net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)

net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)


"""" SECOND CONV LAYER """""
# Input: output from first conv layer
# Filter: 36 filters
# Filter kernel size: 5x5
# Padding: maintains same image size
# ReLU: True
net = tf.layers.conv2d(inputs=net, name='layer_conv2', padding='same',
                       filters=36, kernel_size=5, activation=tf.nn.relu)

net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)


# Flattening the layer from 4d to 2d for the Fully-connected layer
net = tf.layers.flatten(net)


""" FULLY CONNECTED LAYER """
# Input: output from flatten conv layer 2
# Units: number of neurons in the layer
# ReLU: True
net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=128, activation=tf.nn.relu)


"""FINAL CONNECTED LAYER """
# Input: output from first fc layer
# Units: 10 neurons, for each digit
# ReLU: False because we just need the raw output
net = tf.layers.dense(inputs=net, name='layeR_fc2',
                      units=num_classes, activation=None)


logits = net        # Sometimes output of last layer is called the logits

" Squishing final output into numbers between 0 and 1 "
y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, axis=1)


def use_model(input_data, label):
    """
    Method that handles creating a TF Session and getting the results
    of the computation
    Prints it in a human readable array
    :param input_data: the Tensor to input as the input data for the model
    :param label: needed, though unnecessary label input
    :return: None
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, '/tmp/model.ckpt')
    print("Successfully restored ", sess)
    with tf.device("/gpu:0"):
        result = sess.run(y_pred, feed_dict={x: input_data, y_true: label})
        index = np.argmax(result)
        result = [0 for _ in range(10)]
        result[index] = 1
        print(result)
        print(np.argmax(result))
    sess.close()


def sigmoid(num):
    """
    Simple sigmoid function that squishes down input into a number between 0 and 1
    :param num: number to apply the function
    :return: result of sigmoid function
    """
    return 1 / (1 + math.exp(-num))


def main():
    """
    Main method of the program that handles pre-processing the images data
    in order to feed it into the model correct
    Continues until user inputs the exit
    :return: None
    """
    sentinel = True
    while sentinel:
        filename = input("Input file name to check ('q' to quit): ")
        if filename == 'q':
            break
        im = color.rgb2gray(io.imread(filename))
        imarray = np.array(im)
        img = [imarray.flatten()]
        i = 0
        for value in img[0]:
            if img[0][i] == 1:
               img[0][i] = 0
            else:
                img[0][i] = sigmoid(value)
            i+=1
        label = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
        use_model(img, label)

if __name__ == '__main__':
    main()