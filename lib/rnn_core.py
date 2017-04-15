import tensorflow as tf
import numpy as np
import pprint as pp
from abc import abstractmethod


class RNNCore:
    X = None
    Y = None

    _states = None

    hypothesis = None
    cost_function = None

    optimizer = None
    prediction = None

    errors = []
    logs = []

    @abstractmethod
    def init_network(self):
        pass

    @abstractmethod
    def create_writer(self): # for tensorboard
        pass

    @abstractmethod
    def do_summary(self, feed_dict): # for tensorboard
        pass

    def set_placeholder(self, seq_len, hidden_size):
        self.X = tf.placeholder(tf.float32, [None, seq_len, hidden_size])  # None, 6, 5
        self.Y = tf.placeholder(tf.int32, [None, seq_len])  # Y label

    def rnn_lstm_cell(self, hidden_size, batch_size):
        # cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, self._states = tf.nn.dynamic_rnn(cell, self.X, initial_state=initial_state, dtype=tf.float32)

        return outputs

    def set_hypothesis(self, output):
        self.hypothesis = output

    def set_cost_function(self, batch_size, seq_len):
        weights = tf.ones([batch_size, seq_len])
        self.cost_function = tf.contrib.seq2seq.sequence_loss(logits=self.hypothesis, targets=self.Y, weights=weights)
        # sequence_loss = tf.nn.seq2seq.sequence_loss_by_example(logits=outputs, targets=Y, weights=weights)

    def set_optimizer(self, l_rate):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(tf.reduce_mean(self.cost_function))

    def test(self, hihell):
        print('Input:')
        pp.pprint(hihell)
        print('-->')
        ihello = self.sess.run(self.prediction, feed_dict={self.X: hihell})
        print("Hmm, I think.. ", ihello)

        idx2char = ['h', 'i', 'e', 'l', 'o']
        print('-->')
        result_str = [idx2char[c] for c in np.squeeze(ihello)]
        print(''.join(result_str))

    def print_log(self):
        for item in self.logs:
            print(item)

    def learn(self, xonehot, ydata):
        tf.set_random_seed(777)  # reproducibility

        '''
        input_dim = 5  # input dimension
        hidden_size = 5  # output dim. 
        sequence_length = 6  # six inputs
        batch_size = 1   # one sentence
        '''

        self.init_network()

        self.prediction = tf.argmax(self.hypothesis, axis=2)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.create_writer()  # virtual function for tensorboard

        for i in range(100):
            err, _ = self.sess.run([self.cost_function, self.optimizer], feed_dict={self.X: xonehot, self.Y: ydata})

            self.do_summary(feed_dict={self.X: xonehot, self.Y: ydata})  # virtual function for tensorboard

            result = self.sess.run(self.prediction, feed_dict={self.X: xonehot})
            msg = "{} 오류: {:.6f}, 예측: {}, 실제: {}".format(i, err, result, ydata)
            self.logs.append(msg)
