# Lab 6 Softmax Classifier
import tensorflow as tf
from neural_network import NeuralNetwork

from lib.nntype import NNType


class XXX (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(4, 3)

        output = self.fully_connected_layer(self.X, 4, 3, 'W', 'b')
        output = tf.nn.softmax(output)

        self.set_hypothesis(output)
        self.set_cost_function(NNType.SOFTMAX)
        self.set_optimizer(NNType.GRADIENT_DESCENT, 0.1)


x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

gildong = XXX()
gildong.learn(x_data, y_data, 2000, 200)
gildong.test_sigmoid([[1, 11, 7, 9]])
gildong.test_argmax([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]])
#gildong.evaluate_sigmoid(x_data, y_data)


'''
[1, 11, 7, 9]
->
[  1.38904853e-03   9.98601973e-01   9.06129208e-06]
Casted value:  [[ 0.  1.  0.]]

[1, 11, 7, 9]
[1, 3, 4, 3]
[1, 1, 0, 1]
-> Hypothesis
[  1.38904958e-03   9.98601854e-01   9.06129117e-06]
[ 0.93119204  0.06290206  0.0059059 ]
[  1.27327668e-08   3.34112905e-04   9.99665856e-01]
-> tf.arg_max()
[1 0 2]
'''
