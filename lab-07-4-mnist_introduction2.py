# Lab 7 Learning rate and Evaluation
from lib.mnist_neural_network import MnistNeuralNetwork
from lib.nntype import NNType


class XXX (MnistNeuralNetwork):
    def init_network (self):
        self.set_placeholder(784, 10) #28 * 28 = 784, 0~9 digits -> num_of_input, num_of_neuron

        # 784 to 10
        L = self.fully_connected_layer(self.X, 784, 10, 'Wa', 'ba')
        L = self.softmax(L)

        self.set_hypothesis(L)

        self.set_cost_function(NNType.SOFTMAX)
        self.set_optimizer(NNType.GRADIENT_DESCENT, 0.1)


gildong = XXX()
gildong.learn_mnist(15, 100)
gildong.evaluate()
gildong.classify_random()
#gildong.print_log()
gildong.show_error()


'''
Start learning:
...............
Done!

Recognition rate : 0.8942
Label [1]
Classified [3]
'''

