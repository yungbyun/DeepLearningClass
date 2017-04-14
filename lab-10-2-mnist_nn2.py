# Lab 10 MNIST and NN
import tensorflow as tf
from lib.mnist_neural_network import MnistNeuralNetwork
from lib.nntype import NNType


class MyNetwork (MnistNeuralNetwork):
    def init_network(self):
        self.set_placeholder(784, 10)

        L1 = self.create_layer(self.X, 784, 256, 'Wa', 'ba')
        L1 = tf.nn.relu(L1)

        L2 = self.create_layer(L1, 256, 256, 'Wb', 'bb')
        L2 = tf.nn.relu(L2)

        L3 = self.create_layer(L2, 256, 10, 'Wc', 'bc')
        self.set_hypothesis(L3)

        self.set_cost_function(NNType.SOFTMAX_LOGITS)
        self.set_optimizer(NNType.ADAM, 0.001)


gildong = MyNetwork()
gildong.learn_mnist(15, 100)
gildong.evaluate()
gildong.classify_random()
gildong.show_error()

'''
Epoch: 0001 cost = 146.360605907
Epoch: 0002 cost = 40.355669494
Epoch: 0003 cost = 25.141711207
Learning Finished!
Accuracy: 0.91
Label:  [4]
Prediction:  [4]
'''

'''
Epoch: 0013 cost = 0.820965160
Epoch: 0014 cost = 0.624131458
Epoch: 0015 cost = 0.454633765
Learning Finished!
Accuracy: 0.9455
'''
