import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST', one_hot=True)

images = tf.placeholder(tf.float32, [None, 784])
true = tf.placeholder(tf.float32, [None, 10])

image_reshape = tf.reshape(images, shape=[None, 28, 28, 1])
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 4], stddev=0.1))
B1 = tf.Variable(tf.zeros([4]))
W2 = tf.Variable(tf.truncated_normal([4, 4, 4, 8], stddev=0.1))
B2 = tf.Variable(tf.zeros([8]))
W3 = tf.Variable(tf.truncated_normal([4, 4, 8, 12], stddev=0.1))
B3 = tf.Variable(tf.zeros([12]))

W4 = tf.Variable(tf.truncated_normal([7*7*12, 200], stddev=0.1))
B4 = tf.Variable(tf.zeros([200]))

W5 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

l1 = tf.nn.relu(tf.nn.conv2d(image_reshape, W1, strides=[1,1,1,1], padding='SAME') + B1)
l2 = tf.nn.relu(tf.nn.conv2d(l1, W2, strides=[1,2,2,1], padding='SAME') + B2)
l3 = tf.nn.relu(tf.nn.conv2d(l2, W3, strides=[1,2,2,1], padding='SAME') + B3)

reshape = tf.reshape(l3, shape=[-1, 7*7*12])
l4 = tf.nn.relu(tf.matmul(reshape, W4) + B4)

pred = tf.nn.softmax(tf.matmul(l4, W5) + B5)

loss = -tf.reduce_sum(true * tf.log(pred))
optimizer = tf.train.AdamOptimizer(0.005)
train_step = optimizer.minimize(loss)

is_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(true, 1))
acc = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(100):
            x_batch, labels = data.train.next_batch(100)
            sess.run(train_step, feed_dict={images: x_batch, true: labels})

            if _ % 1000 == 0:
                accuracy = sess.run(acc, feed_dict={images: x_batch, true: labels})
                print("Accuracy on iteration", _, ":", accuracy)

        x_batch, labels = data.test.next_batch(9999)
        accuracy = sess.run(acc, feed_dict={images: x_batch, true: labels})
        print("Accuracy on test-set: ", accuracy)