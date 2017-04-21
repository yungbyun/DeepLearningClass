import tensorflow as tf
import numpy as np
import pprint as pp
from abc import abstractmethod
from lib.sentence_to_index import SentenceToIndex
import lib.mytool as mytool

class RNNCore:
    indexing_tool = SentenceToIndex()

    X = None
    Y = None

    _states = None

    hypothesis = None
    cost_function = None

    optimizer = None
    prediction = None

    errors = []
    logs = []

    hidden_size = 0
    output_size = 0
    input_size = 0
    number_of_class = 0
    length_of_sequence = 0
    batch_size = 0

    @abstractmethod
    def init_network(self):
        pass

    @abstractmethod
    def create_writer(self): # for tensorboard
        pass

    @abstractmethod
    def do_summary(self, feed_dict): # for tensorboard
        pass

    def set_parameters(self, sentence):
        tmp = self.indexing_tool.get_unique_char_num(sentence)
        self.input_size = tmp # 유일한 문자 수
        self.hidden_size = tmp # 유일한 문자 수
        self.output_size = tmp # 유일한 문자 수
        self.number_of_class = tmp # 유일한 문자 수
        self.length_of_sequence = len(sentence) - 1 # x_data, y_data 문자 수

    def set_placeholder(self, seq_len, hidden_size):
        self.X = tf.placeholder(tf.int32, [None, seq_len])  # None, 6, 5
        self.Y = tf.placeholder(tf.int32, [None, seq_len])  # Y label

    # 아래 함수는 hypothesis를 정의하므로 learn에서 cost_function, optimizer, predict를 실행하면 결국 이 hypothesis가 실행됨.
    def rnn_lstm_cell2(self, X, num_classes, hidden_size, batch_size):
        # X: [[9, 1, 7, 9, 2, 3, 8, 9, 5, 4, 6, 0, 9, 2, 3]], num_classes: 10
        # X로 입력받는 숫자 각각에 대하여 num_classes 개의 0 중 해당 위치만 1로 만드는 텐서를 리턴함.
        x_one_hot = tf.one_hot(X, num_classes)  # X: 1 -> x_one_hot: 0 1 0 0 0 0 0 0 0 0
        print(x_one_hot) #(?, 15, 10), (?, 6, 5)

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)  # 10
        initial_state = cell.zero_state(batch_size, tf.float32)  # 1
        hypothesis, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)
        print(hypothesis)
        # shape = (1, 15, 10) 글자 하나를 의미하는 출력 벡터가 15개 출력됨.
        # shape = (1, 6, 5)
        return hypothesis

    def set_hypothesis(self, output):
        self.hypothesis = output

    def set_cost_function(self, batch_size, seq_len):
        weights = tf.ones([batch_size, seq_len])
        self.cost_function = tf.contrib.seq2seq.sequence_loss(logits=self.hypothesis, targets=self.Y, weights=weights)
        # sequence_loss = tf.nn.seq2seq.sequence_loss_by_example(logits=outputs, targets=Y, weights=weights)

    def set_optimizer(self, l_rate):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(tf.reduce_mean(self.cost_function))

    def predict(self, xdata):
        print('\nPrediction:')
        print("'{}'".format(xdata), '\n->')
        x_index_list = self.indexing_tool.sentence_to_index_list(xdata)

        prediction = tf.argmax(self.hypothesis, axis=2)
        print(x_index_list)
        index_list = self.sess.run(prediction, feed_dict={self.X: [x_index_list]})
        result_str = self.indexing_tool.index_list_to_sentence(index_list)
        print("'{}'".format(result_str))

    def print_log(self):
        for item in self.logs:
            print(item)

    def learn(self, xd, yd, total_loop, check_step):
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



        x_index_list = self.indexing_tool.sentence_to_index_list(xd)
        #[1, 4, 1, 0, 3, 3]

        #print(self.sess.run(self.hypothesis, feed_dict={self.X: [x_index_list]}))
        '''
        [[
          [-0.0725009 - 0.05906601 - 0.10360178  0.02469962  0.05508834]
          [0.03855447 - 0.00626501 - 0.13088351 - 0.0524451   0.05374868]
          [-0.04780398 - 0.05640227 - 0.19607051 - 0.03313418  0.10015255]
          [-0.04477444 - 0.09467005 - 0.17923516 0.02743356 0.09965682]
          [0.05212242 - 0.04072301 - 0.06027422  0.09040849  0.18318209]
          [0.10590377 0.02277889 - 0.01031762 0.13659562 0.21933225]
         ]
        ] shape=(1,6,5)
        '''
        y_index_list = self.indexing_tool.sentence_to_index_list(yd)
        #[4, 1, 0, 3, 3, 2]

        print('\nStart learning:')

        for i in range(total_loop + 1):
            err, _ = self.sess.run([self.cost_function, self.optimizer],
                    feed_dict={self.X: [x_index_list], self.Y: [y_index_list]})

            self.do_summary(feed_dict={self.X: x_index_list, self.Y: [y_index_list]})  # virtual function for tensorboard

            if i % check_step == 0:
                mytool.print_dot()
                result = self.sess.run(self.prediction, feed_dict={self.X: [x_index_list]})
                msg = "{} 오류: {:.6f}, 예측: {}, 실제: {}".format(i, err, result, [y_index_list])
                self.logs.append(msg)
                self.errors.append(err)

        print('\nDone!\n')

    def show_parameters(self):
        print('hidden_size', self.hidden_size)
        print('output_size', self.output_size)
        print('input_size', self.input_size)
        print('number_of_class', self.number_of_class)
        print('length_of_sequence', self.length_of_sequence)
        print('batch_size', self.batch_size)


    def show_error(self):
        from lib.myplot import MyPlot
        mp = MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.errors)

