'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

'''
사용방법:
gildong = XXX()
mnist = gildong.read_mnist()
gildong.learn(mnist)
gildong.test(mnist)
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


class XXX:
        # Parameters
        learning_rate = 0.01
        training_epochs = 3 #25
        batch_size = 100
        display_step = 1

        # tf Graph Input
        x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
        y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

        # Set model weights
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        # Construct model
        y_prime = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

        # Minimize error using cross entropy
        error = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_prime), reduction_indices=1))
        # Gradient Descent
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(error)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        sess = tf.Session()

        def __init__(self):
            self.sess.run(self.init)

        def read_mnist(self):
            mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
            return mnist

        def learn(self, mnist):
            # Training cycle
            for epoch in range(self.training_epochs):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples / self.batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(self.batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = self.sess.run([self.optimizer, self.error], feed_dict={self.x: batch_xs,
                                                                                  self.y: batch_ys})
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if (epoch + 1) % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

            print("Optimization Finished!")

        def test(self, mnist):
            with self.sess: #이 문장이 없으면 에러가 남. 왜? 예외를 처리하지 못하므로!
                # Test model
                correct_prediction = tf.equal(tf.argmax(self.y_prime, 1), tf.argmax(self.y, 1))
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("Accuracy:", accuracy.eval({self.x: mnist.test.images, self.y: mnist.test.labels}))

        def run(self):
            mnist = self.read_mnist()
            self.learn(mnist)
            self.test(mnist)


