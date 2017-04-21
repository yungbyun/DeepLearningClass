# Lab 12 RNN
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

# under refactoring now... April 21, 2017


class XXX:

    X = None
    Y = None

    errors = []

    char_index_set = []

    hidden_size = 0
    number_of_class = 0
    length_of_sequence = 0
    batch_size = 0

    def string_to_index(self, str):
        x_index_list = [self.char_index_set[c] for c in str]  # x str to index
        return x_index_list

    # 문장 전체(인덱스 리스트)를 xdata, ydata를 위한 부분으로 잘라서 글자(인덱스) 하나씩 오른쪽으로 이동하면서
    # 끝까지 커버하도록. 자른 것들은 dataX, dataY로 저장:
    def cut_and_append(self, sentence):

        dataX = []
        dataY = []

        # 문장전체 1번 학습 위한 반복 횟수
        repeat_time = len(sentence) - self.length_of_sequence # 전체길이: 180, seq_length: 10
        for i in range(0, repeat_time): # i=0 ~ 169 (170번)
            x_sentence_partial = sentence[i:i + self.length_of_sequence] # 0:10, 1:11, 2:12, ...
            y_sentence_partial = sentence[i + 1: i + self.length_of_sequence + 1]
            #print(i, "'{}'".format(x_sentence_partial), '->', "'{}'".format(y_sentence_partial))

            x_index_list_partial = self.string_to_index(x_sentence_partial)  # x str to index
            y_index_list_partial = self.string_to_index(y_sentence_partial)  # y str to index

            dataX.append(x_index_list_partial)
            dataY.append(y_index_list_partial)

        return dataX, dataY

    def set_placeholder(self, seq_len):
        self.X = tf.placeholder(tf.int32, [None, seq_len])  # shape=(?, 10)
        self.Y = tf.placeholder(tf.int32, [None, seq_len])

    def run(self, senten):

        tf.set_random_seed(777)  # reproducibility

        # set -> {}, list -> [], 문장에 있는 단어 중 중복된 것은 제거한 후 리스트로 만듦.
        unique_char_list = list(set(senten))
        self.char_index_set = {w: i for i, w in enumerate(unique_char_list)}

        self.hidden_size = len(unique_char_list)
        self.number_of_class = len(unique_char_list) # 25
        self.length_of_sequence = 10  # Any arbitrary number-> 앞의 예제에서는 x_data, ydata 문자 수(전체 문장 - 1)

        dataX, dataY = self.cut_and_append(senten)
        self.batch_size = len(dataX)

        self.set_placeholder(self.length_of_sequence) # 10

        # create_multi_rnn_layer()
        # One-hot encoding. X: x_data = length_of_sequence 길이의 문자열: 현재로는 10
        X_one_hot = tf.one_hot(self.X, self.number_of_class) # X_one_hot shape=(?, 10, 25)

        # Make a lstm cell with hidden_size (each unit output vector size)
        cell = rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        cell = rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)

        # outputs: unfolding size x hidden size, state = hidden size
        outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32)
        print(outputs) # shape=(?, 10, 25)

        # (optional) softmax layer
        X_for_softmax = tf.reshape(outputs, [-1, self.hidden_size]) #hidden_size = 25
        # flatten the tensor(?, 10, 25). [-1, 25] 25 차원 입력이 되도록 하고 나머지는 flatten
        #print(X_for_softmax) # 따라서 (?, 25)

        # fully connected된 히든 레이어와 출력 레이어 가중치 25 * 25
        softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.number_of_class])
        softmax_b = tf.get_variable("softmax_b", [self.number_of_class])
        outputs2 = tf.matmul(X_for_softmax, softmax_w) + softmax_b

        # reshape outputs for sequence_loss
        outputs2 = tf.reshape(outputs2, [self.batch_size, self.length_of_sequence, self.number_of_class])
        # All weights are 1 (equal weights)
        weights = tf.ones([self.batch_size, self.length_of_sequence])

        sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs2, targets=self.Y, weights=weights)
        mean_loss = tf.reduce_mean(sequence_loss)
        train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(mean_loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for i in range(10):
            _, l, results = sess.run([train_op, mean_loss, outputs2], feed_dict={self.X: dataX, self.Y: dataY})
            for j, result in enumerate(results):
                index = np.argmax(result, axis=1)
                #print(i, j, ''.join([unique_char_list[t] for t in index]), l)
                self.errors.append(l)

        # Let's print the last char of each result to check it works
        results = sess.run(outputs2, feed_dict={self.X: dataX})
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            if j is 0:  # print all for the first result to make a sentence
                print(''.join([unique_char_list[t] for t in index]), end='')
            else:
                print(unique_char_list[index[-1]], end='')

    def show_error(self):
        from lib.myplot import MyPlot
        mp = MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.errors)


gildong = XXX()

sent = ("if you want to build a ship, don't drum up people together to "
        "collect wood and don't assign them tasks and work, but rather "
        "teach them to long for the endless immensity of the sea.")
gildong.run(sent)
gildong.show_error()


'''
0 167 tttttttttt 3.23111
0 168 tttttttttt 3.23111
0 169 tttttttttt 3.23111
…
499 167  of the se 0.229616
499 168 tf the sea 0.229616
499 169   the sea. 0.229616

g you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.

'''
