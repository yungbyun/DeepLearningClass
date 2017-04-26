import tensorflow as tf
from lib.myplot import MyPlot
from abc import abstractmethod
from lib.sentence_to_index import SentenceToIndex


class RNNCore2:
    cheolsu = SentenceToIndex()

    X = None
    Y = None

    hypothesis = None
    cost_function = None
    optimizer = None

    sess = None

    costs = []

    class_size = 0 #10
    sequence_length = 0 #15
    hidden_size = 0 #10
    input_size = 0 #10
    batch_size = 0 #1

    @abstractmethod
    def init_network(self):
        pass

    def set_placeholder(self, seq_len):
        self.X = tf.placeholder(tf.int32, [None, seq_len])  # 15, X data
        self.Y = tf.placeholder(tf.int32, [None, seq_len])  # 15, Y label

    def set_hypothesis(self, hypo):
        self.hypothesis = hypo

    def rnn_lstm_cell(self, X, num_classes, hidden_size, batch_size):
        # X: [[9, 1, 7, 9, 2, 3, 8, 9, 5, 4, 6, 0, 9, 2, 3]], num_classes: 10
        # X로 입력받는 숫자 각각에 대하여 num_classes 개의 0 중 해당 위치만 1로 만드는 텐서를 리턴함.
        x_one_hot = tf.one_hot(X, num_classes)  # X: 1 -> x_one_hot: 0 1 0 0 0 0 0 0 0 0
        print(x_one_hot) # (?, 15, 10) (?, x_data 문자 수, x_data 중복제거 문자 수)

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)  # 10
        initial_state = cell.zero_state(batch_size, tf.float32)  # 1
        hypothesis, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)
        # shape = (1, 15, 10) 글자 하나를 의미하는 출력 벡터가 15개 출력됨.
        return hypothesis

        '''
        oh = tf.one_hot([[9, 1, 7, 9, 2, 3, 8, 9, 5, 4, 6, 0, 9, 2, 3]], 10)

        x_data : ' if you  want yo'
        x_one_hot : 
        [[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.] <- ' '
          [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.] <- i 
          [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.] <- f
          [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.] <- ' '
          [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.] <- y
          [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.] <- o 
          [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.] <- u 
          [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.] <- ' '
          [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.] <- w 
          [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.] <- a
          [ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.] <- n 
          [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.] <- t
          [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.] <- ' '
          [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.] <- y
          [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]]] <- o
        '''

    def set_cost_function(self, batch_size, seq_len):
        weights = tf.ones([batch_size, seq_len])  # shape = (1, 15)
        sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.hypothesis, targets=self.Y, weights=weights)
        loss = tf.reduce_mean(sequence_loss)
        self.cost_function = loss

    def set_optimizer(self, l_rate):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self.cost_function)

    def get_data(self, my_sentence):
        self.cheolsu.make_unique_lists(my_sentence)

        # hyper parameters
        self.hidden_size = len(self.cheolsu.unique_char_and_index)  # 10, RNN output size
        self.num_classes = len(self.cheolsu.unique_char_and_index)  # 10, final output size (RNN or softmax, etc.)

        self.batch_size = 1  # one sample data, one batch
        self.sequence_length = len(my_sentence) - 1  # 16 - 1 = 15, number of lstm rollings (unit #)

        char_index_list = self.cheolsu.sentence_to_index_list(my_sentence)

        x_index_list = [char_index_list[:-1]]  # 가장 끝 문자를 제외한 나머지 문자들의 인덱스 ' if you want yo'의 인덱스 리스트
        y_index_list = [char_index_list[1:]]  # 처음 문자를 제외한 나머지 문자들의 인덱스 'if you want you'의 인덱스 리스트

        x_data = self.cheolsu.index_list_to_sentence(x_index_list)
        y_data = self.cheolsu.index_list_to_sentence(y_index_list)

        return x_data, y_data

    def learn(self, xdata, ydata, total_loop, check_step):
        tf.set_random_seed(777)  # reproducibility

        self.init_network()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('\nStart learning:')

        x_index_list = self.cheolsu.sentence_to_index_list(xdata)
        #[6, 2, 6, 7, 10, 5, 3, 6, 11, 10, 1, 6, 12, 10, 6, 0, 1, 8, 9]

        y_index_list = self.cheolsu.sentence_to_index_list(ydata)
        #[2, 6, 7, 10, 5, 3, 6, 11, 10, 1, 6, 12, 10, 6, 0, 1, 8, 9, 4]

        for i in range(total_loop):  # 3000
            l, _ = self.sess.run([self.cost_function, self.optimizer],
                feed_dict={self.X: [x_index_list], self.Y: [y_index_list]})

            if i % check_step == 0:  # 10
                self.costs.append(l)

                from lib import mytool
                mytool.print_dot()

        print('\nDone!\n')

    def predict(self, xdata):
        print('\nPrediction:')
        print("'{}'".format(xdata), '\n->')
        x_index_list = self.cheolsu.sentence_to_index_list(xdata)

        prediction = tf.argmax(self.hypothesis, axis=2)
        index_list = self.sess.run(prediction, feed_dict={self.X: [x_index_list]})
        result_str = self.cheolsu.index_list_to_sentence(index_list)
        print("'{}'".format(result_str))

    def print_error(self):
        for item in self.costs:
            print(item)

    def show_error(self):
        mp = MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.costs)

