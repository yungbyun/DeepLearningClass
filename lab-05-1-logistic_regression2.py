# Lab 5 Logistic Regression Classifier
import tensorflow as tf
from lib.neural_network import NeuralNetwork
from lib.nntype import NNType


class MyNeuron (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(2, 1)

        # 2 to 1
        output = self.fully_connected_layer(self.X, 2, 1, 'W', 'b')
        output = tf.sigmoid(output)

        self.set_hypothesis(output)
        self.set_cost_function(NNType.LOGISTIC)
        self.set_optimizer(NNType.GRADIENT_DESCENT, l_rate=0.1)


gildong = MyNeuron()

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

gildong.learn(x_data, y_data, 10000, 200);
gildong.show_error()
gildong.test_sigmoid(x_data)
gildong.evaluate_sigmoid(x_data, y_data)
