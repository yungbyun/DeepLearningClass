# Lab 2 Linear Regression
from lib.neural_network_simple2 import NeuralNetworkSimple2


class MyNeuron (NeuralNetworkSimple2):
    def init_network(self):
        self.set_placeholder()

        # 1 to 1
        hypo = self.fully_connected_layer(self.X, 1, 1, 'W', 'b')

        self.set_hypothesis(hypo)
        self.set_cost_function()
        self.set_optimizer(0.01)


gildong = MyNeuron()
gildong.learn([1, 2, 3], [1, 2, 3], 2000, 20)
gildong.learn([1, 2, 3, 4, 5], [2.1, 3.1, 4.1, 5.1, 6.1], 2000, 20)
gildong.test([5])
gildong.test([2.5])
gildong.test([1.5, 3.5])
gildong.show_error()

