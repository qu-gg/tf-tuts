import tensorflow as tf
import numpy as np
from scipy import misc
from PIL import Image
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

### FIRST CONV LAYER
# Input: input image
# Filter: 16 filters with size
# Kernal: 5x5
# Padding: maintains same image size by filling with zeros
# ReLU: True
net = tf.layers.conv2d(inputs=net, name='layer_conv1', padding='same',
                       filters=16, kernel_size=5, activation=tf.nn.relu)

net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)


### SECOND CONV LAYER
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


### FULLY CONNECTED LAYER
# Input: output from flatten conv layer 2
# Units: number of neurons in the layer
# ReLU: True
net = tf.layers.dense(inputs=net, name='layer_fc1',
                      units=128, activation=tf.nn.relu)


### FINAL CONNECTED LAYER
# Input: output from first fc layer
# Units: 10 neurons, for each digit
# ReLU: False because we just need the raw output
net = tf.layers.dense(inputs=net, name='layeR_fc2',
                      units=num_classes, activation=None)


logits = net        # Sometimes output of last layer is called the logits

" Squishing final output into numbers between 0 and 1 "
y_pred = tf.nn.softmax(logits=logits)
y_pred_cls = tf.argmax(y_pred, axis=1)

def rec_label(label):
    label = int(label)
    if label == 0:
        return [1,0,0,0,0,0,0,0,0,0]
    if label == 1:
        return [0,1,0,0,0,0,0,0,0,0]
    if label == 2:
        return [0,0,1,0,0,0,0,0,0,0]
    if label == 3:
        return [0,0,0,1,0,0,0,0,0,0]
    if label == 4:
        return [0,0,0,0,1,0,0,0,0,0]
    if label == 5:
        return [0,0,0,0,0,1,0,0,0,0]
    if label == 6:
        return [0,0,0,0,0,0,1,0,0,0]
    if label == 7:
        return [0,0,0,0,0,0,0,1,0,0]
    if label == 8:
        return [0,0,0,0,0,0,0,0,1,0]
    if label == 9:
        return [0,0,0,0,0,0,0,0,0,1]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def use_model(input_data, label):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()
    saver.restore(sess, '/tmp/model.ckpt')
    print("Succesfully restored ", sess)
    with tf.device("/gpu:0"):
        result = sess.run(y_pred, feed_dict={x: input_data, y_true: label})
        highest_index = 0
        highest = 0
        i = 0
        for value in result[0]:
            if value > highest:
                highest = value
                highest_index = i
            i+=1
        print(result)
        print(highest_index)
    sess.close()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def main():
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
        label = [[0,0,0,0,0,0,0,0,0,1]]
        use_model(img, label)

if __name__ == '__main__':
    main()