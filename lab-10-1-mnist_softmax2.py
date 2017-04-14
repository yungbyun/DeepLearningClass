# Lab 7 Learning rate and Evaluation
from lib.mnist_neural_network import MnistNeuralNetwork
from lib.nntype import NNType


class MyNeuron (MnistNeuralNetwork):
    def init_network(self):
        self.set_placeholder(784, 10)

        # 784 to 10
        logit = self.fully_connected_layer(self.X, 784, 10, 'Wa', 'ba')  # for logits

        self.set_hypothesis(logit)
        self.set_cost_function(NNType.SOFTMAX_LOGITS)
        self.set_optimizer(NNType.ADAM, 0.001)


gildong = MyNeuron()
gildong.learn_mnist(15, 100)
gildong.evaluate()
#gildong.classify_random()
#gildong.show_error()


'''
Epoch: 0001 cost = 5.916487225
Epoch: 0002 cost = 1.868882108
Epoch: 0003 cost = 1.165184578
Epoch: 0004 cost = 0.895758619
Epoch: 0005 cost = 0.753617214
Epoch: 0006 cost = 0.665425729
Epoch: 0007 cost = 0.604189989
Epoch: 0008 cost = 0.559039789
Epoch: 0009 cost = 0.523200151
Epoch: 0010 cost = 0.494997439
Epoch: 0011 cost = 0.471757663
Epoch: 0012 cost = 0.451736393
Epoch: 0013 cost = 0.435549789
Epoch: 0014 cost = 0.421230042
Epoch: 0015 cost = 0.408108964
Learning Finished!
Accuracy: 0.9014
'''
