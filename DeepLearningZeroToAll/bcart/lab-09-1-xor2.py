# Lab 9 XOR
# This example does not work
import tensorflow as tf
from neural_network import NeuralNetwork

from lib.nntype import NNType


class XXX (NeuralNetwork) :
    def init_network(self):
        self.set_placeholder(2, 1)

        L = self.fully_connected_layer(self.X, 2, 1, 'W', 'b')
        L = tf.nn.sigmoid(L)

        #self.set_weight_bias(2, 1)
        self.set_hypothesis(L)

        self.set_cost_function(NNType.LOGISTIC)
        self.set_optimizer(NNType.GRADIENT_DESCENT, 0.1)

    def my_log(self, i, x_data, y_data):
        super().my_log(i, x_data, y_data)


gildong = XXX()
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [1], [1], [0]]
gildong.learn(x_data, y_data, 4000, 100)
gildong.evaluate_sigmoid(x_data, y_data)
gildong.show_error()
#gildong.print_log()


'''
4000 0.693147

Hypothesis:  [[ 0.5]
 [ 0.5]
 [ 0.5]
 [ 0.5]]
Correct:  [[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
Accuracy:  0.5
'''
