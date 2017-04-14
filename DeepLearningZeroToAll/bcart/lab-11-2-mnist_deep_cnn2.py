# Lab 11 MNIST and Deep learning CNN
import tensorflow as tf

from lib.dropout_mnist_cnn import DropoutMnistCNN


class XXXModel (DropoutMnistCNN):
    def init_network(self):
        self.set_placeholder(784, 10, 28, 28)

        # 1, 2
        CL_a = self.convolution_layer(self.X_2d, 3, 3, 1, 32, 1, 1)
        CL_a = self.relu(CL_a)

        CL_a_maxp = self.max_pool(CL_a, 2, 2, 2, 2)
        CL_a_maxp = self.dropout(CL_a_maxp)

        # 3, 4
        CL_b = self.convolution_layer(CL_a_maxp, 3, 3, 32, 64, 1, 1)
        CL_b = self.relu(CL_b)

        CL_b_maxp = self.max_pool(CL_b, 2, 2, 2, 2)
        CL_b_maxp = self.dropout(CL_b_maxp)

        # 5, 6
        CL_c = self.convolution_layer(CL_b_maxp, 3, 3, 64, 128, 1, 1)
        CL_c = self.relu(CL_c)

        CL_c_maxp = self.max_pool(CL_c, 2, 2, 2, 2)
        CL_c_maxp = self.dropout(CL_c_maxp)

        # 7
        reshaped = tf.reshape(CL_c_maxp, [-1, 128 * 4 * 4])
        output = self.fully_connected_layer(reshaped, 128 * 4 * 4, 625, 'W4')
        L4 = self.relu(output)
        L4 = self.dropout(L4)

        # 8
        hypo = self.fully_connected_layer(L4, 625, 10, 'W5')
        self.set_hypothesis(hypo)

        self.set_cost_function()
        self.set_optimizer(0.001)


gildong = XXXModel()
gildong.learn_mnist(1, 100)
gildong.evaluate()
gildong.classify_random()


'''
Learning stared. It takes sometime.
Epoch: 0001 cost = 0.385748474
Epoch: 0002 cost = 0.092017397
Epoch: 0003 cost = 0.065854684
Epoch: 0004 cost = 0.055604566
Epoch: 0005 cost = 0.045996377
Epoch: 0006 cost = 0.040913645
Epoch: 0007 cost = 0.036924479
Epoch: 0008 cost = 0.032808939
Epoch: 0009 cost = 0.031791007
Epoch: 0010 cost = 0.030224456
Epoch: 0011 cost = 0.026849916
Epoch: 0012 cost = 0.026826763
Epoch: 0013 cost = 0.027188021
Epoch: 0014 cost = 0.023604777
Epoch: 0015 cost = 0.024607201
Learning Finished!
Accuracy: 0.9938
'''
