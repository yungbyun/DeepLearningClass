'''
A nearest neighbor learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


class XXX:
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # In this example, we limit mnist data
    Xtr, Ytr = mnist.train.next_batch(500)  # 5000 for training (nn candidates)
    Xte, Yte = mnist.test.next_batch(20)  # 200 for testing

    # tf Graph Input
    xtr = tf.placeholder("float", [None, 784])
    xte = tf.placeholder("float", [784])

    # Nearest Neighbor calculation using L1 Distance
    # Calculate L1 Distance
    distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
    # Prediction: Get min distance index (Nearest neighbor)
    pred = tf.arg_min(distance, 0)

    sess = 0

    def __init__(self):
        self.sess = tf.Session()

        # Initializing the variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def run(self):
        accuracy = 0.
        # Launch the graph
        with self.sess:
            # loop over test data
            for i in range(len(self.Xte)):
                # Get nearest neighbor
                nn_index = self.sess.run(self.pred, feed_dict={self.xtr: self.Xtr, self.xte: self.Xte[i, :]})
                # Get nearest neighbor class label and compare it to its true label
                print("Test", i, "Prediction:", np.argmax(self.Ytr[nn_index]), \
                    "True Class:", np.argmax(self.Yte[i]))
                # Calculate accuracy
                if np.argmax(self.Ytr[nn_index]) == np.argmax(self.Yte[i]):
                    accuracy += 1./len(self.Xte)
            print("Done!")
            print("Accuracy:", accuracy)


gildong = XXX()
gildong.run()

