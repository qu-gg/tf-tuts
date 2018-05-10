import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
true = tf.placeholder(tf.float32, [None, 10])
k, l, m, n = 200, 100, 60, 30

w1 = tf.Variable(tf.truncated_normal([28*28, k], stddev=0.1))
b1 = tf.Variable(tf.zeros([k]))
w2 = tf.Variable(tf.truncated_normal([k, l], stddev=0.1))
b2 = tf.Variable(tf.zeros([l]))
w3 = tf.Variable(tf.truncated_normal([l, m], stddev=0.1))
b3 = tf.Variable(tf.zeros([m]))
w4 = tf.Variable(tf.truncated_normal([m, n], stddev=0.1))
b4 = tf.Variable(tf.zeros([n]))
w5 = tf.Variable(tf.truncated_normal([n, 10], stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))

x_reshape = tf.reshape(x, [-1, 784])
y1 = tf.nn.relu(tf.matmul(x_reshape, w1) + b1)
y2 = tf.nn.relu(tf.matmul(y1, w2) + b2)
y3 = tf.nn.relu(tf.matmul(y2, w3) + b3)
y4 = tf.nn.relu(tf.matmul(y3, w4) + b4)
pred = tf.nn.softmax(tf.matmul(y4, w5) + b5)


loss = -tf.reduce_sum(true * tf.log(pred))
optimizer = tf.train.AdamOptimizer(0.005)
train_step = optimizer.minimize(loss)

is_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(true, 1))
acc = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(100):
            x_batch, labels = data.train.next_batch(100)
            sess.run(train_step, feed_dict={x: x_batch, true: labels})

            if _ % 1000 == 0:
                accuracy = sess.run(acc, feed_dict={x: x_batch, true: labels})
                print("Accuracy on iteration", _, ":", accuracy)

        x_batch, labels = data.test.next_batch(9999)
        accuracy = sess.run(acc, feed_dict={x: x_batch, true: labels})
        print("Accuracy on test-set: ", accuracy)