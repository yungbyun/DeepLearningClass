import tensorflow as tf
from file2buffer import File2Buffer
from neural_network import NeuralNetwork

from lib.nntype import NNType


class SoftmaxOnehot (NeuralNetwork):
    Y_one_hot = None

    def set_placeholder(self, num_of_input, num_of_output):
        self.X = tf.placeholder(tf.float32, [None, num_of_input])
        self.Y = tf.placeholder(tf.int32, [None, num_of_output])

    #num_of_class: 7, if self.Y is 4 then generates [[0],[0],[0],[0],[1],[0],[0]] as Y_one_hot
    def set_one_hot(self, num_of_class):
        self.Y_one_hot = tf.one_hot(self.Y, num_of_class)  # one hot
        print("one_hot", self.Y_one_hot)
        self.Y_one_hot = tf.reshape(self.Y_one_hot, [-1, num_of_class]) #리스트 [[a],[b]] -> [a, b]
        print("reshape", self.Y_one_hot)

    def create_layer(self, previous_output, num_of_input, num_of_neuron, w_name='weight', b_name='bias'):
        self.set_weight_initializer() ## a hole for you

        if self.xaiver_initializer == NNType.XAIVER:
            # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            W = tf.get_variable(w_name, shape=[num_of_input, num_of_neuron], initializer = tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([num_of_neuron]), name=b_name)
        else : # if self.initializer == None:
            W = tf.Variable(tf.random_normal([num_of_input, num_of_neuron]), name = w_name)
            b = tf.Variable(tf.random_normal([num_of_neuron]), name = b_name)

        # tf.nn.softmax computes softmax activations
        # softmax = exp(logits) / reduce_sum(exp(logits), dim)
        logits = tf.matmul(previous_output, W) + b
        return logits

    def set_hypothesis(self, logits):
        self.hypothesis = tf.nn.softmax(logits)

    def set_cost_function(self, logits):
        cost_i = tf.nn.softmax_cross_entropy_with_logits(logits, labels=self.Y_one_hot)
        self.cost_function = tf.reduce_mean(cost_i)

    def evaluate(self, afile):
        f2b = File2Buffer()
        f2b.file_load(afile)

        index_of_max_value = tf.argmax(self.hypothesis, 1)
        hit_record = tf.equal(index_of_max_value, tf.argmax(self.Y_one_hot, 1))
        recognition_rate = tf.reduce_mean(tf.cast(hit_record, tf.float32))
        #
        acc = self.sess.run(recognition_rate, feed_dict={self.X: f2b.x_data, self.Y: f2b.y_data})
        print("Acc: {:.2%}".format(acc*100))



