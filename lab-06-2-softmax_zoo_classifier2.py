# Lab 6 Softmax Classifier
from lib.neural_network_one_hot import NeuralNetworkOneHot
from lib.nntype import NNType


class XXX (NeuralNetworkOneHot):
    def init_network(self):
        self.set_placeholder(16, 1)

        self.target_to_one_hot(7)

        # 16 to 7
        logits = self.fully_connected_layer(self.X, 16, 7, 'W', 'b')
        hypothesis = self.softmax(logits)

        self.set_hypothesis(hypothesis)
        self.set_cost_function_with_one_hot(logits, self.get_one_hot()) #not hypothesis, but logits
        self.set_optimizer(NNType.GRADIENT_DESCENT, 0.1)


gildong = XXX()
xdata, ydata = gildong.load_file('data-04-zoo.csv')
gildong.learn(xdata, ydata, 2000, 100)
gildong.print_error()
gildong.evaluate('data-04-zoo.csv')
gildong.show_error()

