# Lab 5 Logistic Regression Classifier
import tensorflow as tf
from lib.file2buffer import File2Buffer
from lib.neural_network import NeuralNetwork
from lib.nntype import NNType


class XXX (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(8, 1)

        # 8 to 1
        output = self.fully_connected_layer(self.X, 8, 1, 'W', 'b')
        output = tf.sigmoid(output)

        self.set_hypothesis(output)
        self.set_cost_function(NNType.LOGISTIC)
        self.set_optimizer(NNType.GRADIENT_DESCENT, l_rate=0.01)

    def my_log(self, i, x_data, y_data):
        pass


gildong = XXX()
gildong.learn_with_file('data-03-diabetes.csv', 10000, 200) #10000, 200
gildong.test_sigmoid([[0.176471,0.155779,0,0,0,0.052161,-0.952178,-0.733333]])

f2b = File2Buffer()
f2b.file_load('data-03-diabetes.csv')
gildong.evaluate_sigmoid(f2b.x_data, f2b.y_data)
gildong.show_error()

