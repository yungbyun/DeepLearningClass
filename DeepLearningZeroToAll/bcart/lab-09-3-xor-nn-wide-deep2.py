# Lab 9 XOR
# This example does not work
import numpy as np
import tensorflow as tf
from neural_network import NeuralNetwork

from lib.nntype import NNType


class XXX (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(2, 1)

        L1 = self.fully_connected_layer(self.X, 2, 10, 'Wa', 'ba')  # input
        L1 = tf.sigmoid(L1)

        L2 = self.fully_connected_layer(L1, 10, 10, 'Wb', 'bb')  # hidden1
        L2 = tf.sigmoid(L2)

        L3 = self.fully_connected_layer(L2, 10, 10, 'Wc', 'bc')  # hidden2
        L3 = tf.sigmoid(L3)

        L4 = self.fully_connected_layer(L3, 10, 1, 'Wd', 'bd')  # output
        L4 = tf.sigmoid(L4)

        self.set_hypothesis(L4)

        self.set_cost_function(NNType.LOGISTIC)
        self.set_optimizer(NNType.GRADIENT_DESCENT, 0.1)


gildong = XXX()
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
ydata = np.array([[0], [1], [1], [0]], dtype=np.float32)
gildong.learn(xdata, ydata, 2000, 100)
gildong.evaluate_sigmoid(xdata, ydata)
gildong.show_error()


'''
2000
Hypothesis:  [[ 0.06918658]
 [ 0.92086309]
 [ 0.86154044]
 [ 0.14349058]]
Correct:  [[ 0.]
 [ 1.]
 [ 1.]
 [ 0.]]
Accuracy:  1.0
'''
