# Lab 12 Character Sequence RNN
import tensorflow as tf
import numpy as np
from lib.myplot import MyPlot

# under refactoring...

class SentenceToIndex:
    unique_char_list = []
    unique_char_and_index = []

    def set_sentence(self, sentence):
        unique_char_collec = set(my_sentence)  # set class는 중복된 문자(space 3개, y, o, u)를 제거한 후 무작위로 collection 생성
        # tmp = {'n', 't', 'y', 'w', ' ', 'f', 'u', 'a', 'i', 'o'}

        self.unique_char_list = list(unique_char_collec)  # index -> char
        # uique_char_list = ['n', 't', 'y', 'w', ' ', 'f', 'u', 'a', 'i', 'o']

        aa = enumerate(self.unique_char_list)

        # {'y': 0, 'a': 1, 'f': 2, 'o': 3, 'i': 4, 'w': 5, 't': 6, 'u': 7, 'n': 8, ' ': 9}
        self.unique_char_and_index = {c: i for i, c in aa}
        #print(unique_char_and_index)

    def index_to_sentence(self, index_list):
        str = [self.unique_char_list[c] for c in np.squeeze(index_list)]
        return str


class XXX:
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

    def set_placeholder(self, seq_len):
        self.X = tf.placeholder(tf.int32, [None, seq_len])  # 15, X data
        self.Y = tf.placeholder(tf.int32, [None, seq_len])  # 15, Y label

    def set_hypothesis(self, hypo):
        self.hypothesis = hypo

    def rnn_lstm_cell(self, X, num_classes, hidden_size, batch_size):
        # X: [[9, 1, 7, 9, 2, 3, 8, 9, 5, 4, 6, 0, 9, 2, 3]], num_classes: 10
        # X로 입력받는 숫자 각각에 대하여 num_classes 개의 0 중 해당 위치만 1로 만드는 텐서를 리턴함.
        x_one_hot = tf.one_hot(X, num_classes)  # X: 1 -> x_one_hot: 0 1 0 0 0 0 0 0 0 0

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
        weights = tf.ones([batch_size, seq_len]) #shape = (1, 15)
        sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.hypothesis, targets=self.Y, weights=weights)
        loss = tf.reduce_mean(sequence_loss)
        self.cost_function = loss

    def set_optimizer(self, l_rate):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self.cost_function)

    def sentence_to_data(self, my_sentence):
        self.cheolsu.set_sentence(my_sentence)

        # hyper parameters
        self.hidden_size = len(self.cheolsu.unique_char_and_index)  # 10, RNN output size
        self.num_classes = len(self.cheolsu.unique_char_and_index)  # 10, final output size (RNN or softmax, etc.)

        self.batch_size = 1  # one sample data, one batch
        self.sequence_length = len(my_sentence) - 1  # 16 - 1 = 15, number of lstm rollings (unit #)

        # 샘플 문장에 있는 문자 순서대로 인덱스를 구함
        # ' if you want you' 문장 전체에 있는 문자 인덱스 리스트
        char_index_list = [self.cheolsu.unique_char_and_index[c] for c in my_sentence]  # char to index
        # [7, 1, 3, 7, 6, 5, 9, 7, 8, 0, 4, 2, 7, 6, 5, 9]

        x_data = [char_index_list[:-1]]  # 가장 끝 문자를 제외한 나머지 문자들의 인덱스 ' if you want yo'의 인덱스 리스트
        y_data = [char_index_list[1:]]   # 처음 문자를 제외한 나머지 문자들의 인덱스 'if you want you'의 인덱스 리스트

        return x_data, y_data

    def init_network(self):
        self.set_placeholder(self.sequence_length)

        hypothesis = self.rnn_lstm_cell(self.X, self.num_classes, self.hidden_size, self.batch_size)

        self.set_hypothesis(hypothesis)
        self.set_cost_function(self.batch_size, self.sequence_length)
        self.set_optimizer(0.1)

    def learn(self, xdata, ydata, total_loop, check_step):
        tf.set_random_seed(777)  # reproducibility

        self.init_network()
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('\nStart learning:')

        for i in range(total_loop): #3000
            l, _ = self.sess.run([self.cost_function, self.optimizer], feed_dict={self.X: xdata, self.Y: ydata})

            if i % check_step == 0: #10
                self.costs.append(l)

                from lib import mytool
                mytool.print_dot()

        print('\nDone!\n')


    def predict(self, xdata):
        prediction = tf.argmax(self.hypothesis, axis=2)
        result = self.sess.run(prediction, feed_dict={self.X: xdata})
        result_str = self.cheolsu.index_to_sentence(result)
        print("Prediction:", ''.join(result_str))

    def print_error(self):
        for item in self.costs:
           print(item)

    def show_error(self):
        mp = MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.costs)


gildong = XXX()
my_sentence = " if you want you"
x_data, y_data = gildong.sentence_to_data(my_sentence)
gildong.learn(x_data, y_data, 200, 20) #3000
gildong.print_error()
gildong.predict(x_data)



'''
0 loss: 2.29895 Prediction: nnuffuunnuuuyuy
1 loss: 2.29675 Prediction: nnuffuunnuuuyuy
2 loss: 2.29459 Prediction: nnuffuunnuuuyuy
3 loss: 2.29247 Prediction: nnuffuunnuuuyuy

...

1413 loss: 1.3745 Prediction: if you want you
1414 loss: 1.3743 Prediction: if you want you
1415 loss: 1.3741 Prediction: if you want you
1416 loss: 1.3739 Prediction: if you want you
1417 loss: 1.3737 Prediction: if you want you
1418 loss: 1.37351 Prediction: if you want you
1419 loss: 1.37331 Prediction: if you want you
'''
