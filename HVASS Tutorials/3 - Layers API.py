import tensorflow as tf
import numpy as np
import math
from tensorflow.examples.tutorials.mnist import input_data
device = "/gpu:0"


data = input_data.read_data_sets('data/MNIST', one_hot=True)
data.test.cls = np.argmax(data.test.labels, axis=1)

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

"""
Cost Function + Optimizer
"""

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)

cost = tf.reduce_mean(cross_entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=0.00085).minimize(cost)

"""
Accuracy
"""
correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


"""
Run Session
"""

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.Saver()

with tf.device(device):
    sess.run(tf.global_variables_initializer())

    train_batch_size = 64

    total_iterations = 0

    def optimize(num_iterations):
        global total_iterations

        for i in range(total_iterations, total_iterations + num_iterations):
            x_batch, y_true_batch = data.train.next_batch(train_batch_size)

            feed_dict_train = {x: x_batch,
                               y_true: y_true_batch}

            sess.run(optimizer, feed_dict=feed_dict_train)

            if i % 100 == 0:
                acc = sess.run(accuracy, feed_dict=feed_dict_train)

                msg = "Opt Iteration: {0:>6}, Accuracy: {1:>6.1%}"
                print(msg.format(i + 1, acc))

        total_iterations += num_iterations


    test_batch_size = 256
    def print_acc():
        num_test = len(data.test.images)
        cls_pred = np.zeros(shape=num_test, dtype=np.int)

        i = 0
        while i < num_test:
            j = min(i + test_batch_size, num_test)

            images = data.test.images[i:j, :]
            labels = data.test.labels[i:j, :]

            feed_dict = {x: images,
                         y_true: labels}

            cls_pred[i:j] = sess.run(y_pred_cls, feed_dict=feed_dict)

            i = j

        cls_true = data.test.cls
        correct = (cls_true == cls_pred)
        correct_sum = correct.sum()

        acc = float(correct_sum) / num_test
        msg = "Acc on Test-Set: {0:.1%} ({1}/{2})"
        print(msg.format(acc, correct_sum, num_test))

    optimize(10000)
    print_acc()
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in path: %s" % save_path)

sess.close()
