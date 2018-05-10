import os
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data

data = input_data.read_data_sets('data/MNIST', one_hot=True)

images = tf.placeholder(tf.float32, [None, 784])
true = tf.placeholder(tf.float32, [None, 10])

f = 6
s = 12
t = 24

image_reshape = tf.reshape(images, shape=[-1, 28, 28, 1])
W1 = tf.Variable(tf.truncated_normal([6, 6, 1, f], stddev=0.1), name='w1')
B1 = tf.Variable(tf.ones([f])/10)
W2 = tf.Variable(tf.truncated_normal([5, 5, f, s], stddev=0.1))
B2 = tf.Variable(tf.ones([s])/10)
W3 = tf.Variable(tf.truncated_normal([4, 4, s, t], stddev=0.1))
B3 = tf.Variable(tf.ones([t])/10)

W4 = tf.Variable(tf.truncated_normal([7*7*t, 200], stddev=0.1))
B4 = tf.Variable(tf.ones([200])/10)

W5 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
B5 = tf.Variable(tf.ones([10])/10)

l1 = tf.nn.relu(tf.nn.conv2d(image_reshape, W1, strides=[1, 1, 1, 1], padding='SAME') + B1)
l2 = tf.nn.relu(tf.nn.conv2d(l1, W2, strides=[1, 2, 2, 1], padding='SAME') + B2)
l3 = tf.nn.relu(tf.nn.conv2d(l2, W3, strides=[1, 2, 2, 1], padding='SAME') + B3)

reshape = tf.reshape(l3, shape=[-1, 7*7*t])

drop_rate = tf.placeholder(tf.float32)
dropped = tf.nn.dropout(reshape, keep_prob=drop_rate)

l4 = tf.nn.relu(tf.matmul(dropped, W4) + B4)

pred = tf.nn.softmax(tf.matmul(l4, W5) + B5)

loss = -tf.reduce_sum(true * tf.log(pred))
optimizer = tf.train.AdamOptimizer(0.005)
train_step = optimizer.minimize(loss)

is_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(true, 1))
acc = tf.reduce_mean(tf.cast(is_correct, tf.float32))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(2000):
            x_batch, labels = data.train.next_batch(100)
            sess.run(train_step, feed_dict={images: x_batch, true: labels, drop_rate: 0.70})

            if _ % 500 == 0:
                accuracy = sess.run(acc, feed_dict={images: x_batch, true: labels, drop_rate: 0.70})
                print("Accuracy on iteration", _, ":", accuracy)

        x_batch, labels = data.test.next_batch(9999)
        accuracy = sess.run(acc, feed_dict={images: x_batch, true: labels, drop_rate: 1.0})
        print("Accuracy on test-set: ", accuracy)