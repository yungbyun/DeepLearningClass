from lib.neural_network_simple1 import NeuralNetworkSimple1


class MyNeuron (NeuralNetworkSimple1):
    def init_network(self):

        # 1 to 1
        hypo = self.perceptron(1, 1, 'W', 'b') # hypo = W*x + b

        self.set_hypothesis(hypo)
        self.set_cost_function()
        self.set_optimizer(0.1)


gildong = MyNeuron()

x_data =[1, 2, 3]
y_data = [1, 2, 3]

gildong.set_data(x_data, y_data)
gildong.learn(2000, 50)
gildong.test()
gildong.show_error()



