# Lab 4 Multi-variable linear regression

from neural_network import NeuralNetwork

from lib.nntype import NNType


class LinearRegressionFromFile (NeuralNetwork):
    def init_network(self):
        self.set_placeholder(3, 1)

        output = self.fully_connected_layer(self.X, 3, 1, 'W', 'b')

        self.set_hypothesis(output)
        self.set_cost_function(NNType.SQUARE_MEAN)
        self.set_optimizer(NNType.GRADIENT_DESCENT, l_rate=1e-5)

    def my_log(self, i, x_data, y_data):
        pass


gildong = LinearRegressionFromFile()
gildong.learn_with_file('data-01-test-score.csv', 2000, 10)
gildong.test_linear([[100, 70, 101]])
gildong.test_linear([[60, 70, 110], [90, 100, 80]])

'''
[100, 70, 101]
->
[ 181.73277283]

[60, 70, 110]
[90, 100, 80]
->
[ 145.86265564]
[ 187.23130798]
'''