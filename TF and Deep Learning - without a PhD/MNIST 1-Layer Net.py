import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.matmul(tf.reshape(x, [-1, 784]), weights) + biases)
true = tf.placeholder(tf.float32, [None, 10])

loss = -tf.reduce_sum(true * tf.log(pred))
optimizer = tf.train.AdamOptimizer(0.005)
train_step = optimizer.minimize(loss)

is_correct = tf.equal(tf.argmax(pred, 1), tf.argmax(true, 1))
acc = tf.reduce_mean(tf.cast(is_correct, tf.float32))

with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(1000):
            x_batch, labels = data.train.next_batch(100)
            sess.run(train_step, feed_dict={x: x_batch, true: labels})

            if _ % 100 == 0:
                accuracy = sess.run(acc, feed_dict={x: x_batch, true: labels})
                print("Accuracy on iteration", _, ":", accuracy)

