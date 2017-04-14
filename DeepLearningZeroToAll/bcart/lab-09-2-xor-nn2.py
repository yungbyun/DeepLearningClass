# Lab 9 XOR
# This example does not work
import numpy as np
import tensorflow as tf
from neural_network import NeuralNetwork

from lib.nntype import NNType


class XXX (NeuralNetwork) :
    def init_network(self):
        self.set_placeholder(2, 1)

        L1 = self.fully_connected_layer(self.X, 2, 2, 'Wa', 'ba')
        L1 = tf.sigmoid(L1)

        L2 = self.fully_connected_layer(L1, 2, 1, 'Wb', 'bb')
        L2 = tf.sigmoid(L2)

        self.set_hypothesis(L2)

        self.set_cost_function(NNType.LOGISTIC)
        self.set_optimizer(NNType.GRADIENT_DESCENT, 0.1)

    def my_log(self, i, xdata, ydata):
        pass


gildong  = XXX()
xdata = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
ydata = np.array([[0], [1], [1], [0]], dtype=np.float32)
gildong.learn(xdata, ydata, 3000, 100)
gildong.evaluate_sigmoid(xdata, ydata)
gildong.test_sigmoid([[0, 1], [1, 0]])
gildong.show_error()


'''
when the loop is 3001,
Hypothesis:  [[ 0.18309775]
 [ 0.72479963]
 [ 0.8890698 ]
 [ 0.13911283]]
Correct:  [[ 0.]
 [ 1.]
 [ 1.]
 [ 0.]]
Accuracy:  1.0
'''
