import tensorflow as tf

from lib.neural_network import NeuralNetwork

'''
Example of network initialization
NOTE: Don't forget to setup an op for one_hot encoding from target values

def init_network(self):
    self.set_placeholder(16, 1)

    self.target_to_one_hot(7)

    logits = self.create_layer(self.X, 16, 7, 'W', 'b')
    hypothesis = self.softmax(logits)

    self.set_hypothesis(hypothesis)
    self.set_cost_function_with_one_hot(logits, self.get_one_hot()) #not hypothesis, but logits
    self.set_optimizer(NNType.GRADIENT_DESCENT, 0.1)
'''


class NeuralNetworkOneHot (NeuralNetwork):
    Y = None
    Y_one_hot_reshaped = None

    def get_one_hot(self):
        return self.Y_one_hot_reshaped

    def set_placeholder(self, num_of_input, num_of_output):
        self.X = tf.placeholder(tf.float32, [None, num_of_input])
        self.Y = tf.placeholder(tf.int32, [None, num_of_output])  # 0 ~ 6

    def set_cost_function_with_one_hot(self, logits, reshaped):
        # Cross entropy cost/loss
        cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=reshaped)
        cost = tf.reduce_mean(cost_i)
        self.cost_function = cost

    def evaluate(self, afile):
        x_data, y_data = self.load_file(afile)

        prediction = tf.argmax(self.hypothesis, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(self.Y_one_hot_reshaped, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = self.sess.run(accuracy, feed_dict={self.X: x_data, self.Y: y_data})
        print("Acc: {:.2%}".format(acc))

    def target_to_one_hot(self, num_of_class):
        # define an op to get one_hot incoding from (a) target value(Y)
        Y_one_hot2 = tf.one_hot(self.Y, 7)  # one hot op
        self.Y_one_hot_reshaped = tf.reshape(Y_one_hot2, [-1, 7]) # reshape op, 리스트 [[a],[b]] -> [a, b]
