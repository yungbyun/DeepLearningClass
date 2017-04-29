from lib.sentence_tool import SentenceTool
from abc import abstractmethod
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from lib.mytool import print_dot

# under refactoring now...
class SentenceMultiRNN:
    tool = SentenceTool()

    X = None
    Y = None

    hypothesis = None
    cost_function = None
    optimizer = None

    errors = []

    sess = None

    # parameters
    hidden_size = 0
    number_of_class = 0
    length_of_sequence = 0
    batch_size = 0

    @abstractmethod
    def init_network(self):
        pass

    def set_placeholder(self, seq_len):
        self.X = tf.placeholder(tf.int32, [None, seq_len])  # shape=(?, 10)
        self.Y = tf.placeholder(tf.int32, [None, seq_len])

    def create_multi_rnn_layer(self):
        # create_multi_rnn_layer()
        # One-hot encoding. X: x_data = length_of_sequence 길이의 문자열: 현재로는 10
        X_one_hot = tf.one_hot(self.X, self.number_of_class)  # X_one_hot shape=(?, 10, 25)

        # Make a lstm cell with hidden_size (each unit output vector size)
        cell = rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        cell = rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)

        # outputs: unfolding size x hidden size, state = hidden size
        outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, dtype=tf.float32)
        # print(outputs) # shape=(?, 10, 25)

        # (optional) softmax layer
        X_for_softmax = tf.reshape(outputs, [-1, self.hidden_size])  # hidden_size = 25
        # flatten the tensor(?, 10, 25). [-1, 25] 25 차원 입력이 되도록 하고 나머지는 flatten
        #print(X_for_softmax) # 따라서 (?, 25)

        # fully connected된 히든 레이어와 출력 레이어 가중치 25 * 25, 바이어스 25
        softmax_w = tf.get_variable("softmax_w", [self.hidden_size, self.number_of_class])
        softmax_b = tf.get_variable("softmax_b", [self.number_of_class])
        hypothesis = tf.matmul(X_for_softmax, softmax_w) + softmax_b

        # reshape outputs for sequence_loss
        hypothesis_reshaped = tf.reshape(hypothesis, [self.batch_size, self.length_of_sequence, self.number_of_class])
        # All weights are 1 (equal weights)

        return hypothesis_reshaped

    def set_hypothesis(self, hypo):
        self.hypothesis = hypo

    def set_cost_function(self):
        weights = tf.ones([self.batch_size, self.length_of_sequence])
        sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=self.hypothesis, targets=self.Y, weights=weights)
        mean_loss = tf.reduce_mean(sequence_loss)
        self.cost_function = mean_loss

    def set_optimizer(self, l_rate):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self.cost_function)

    def set_parameters(self, s, len_of_seq):
        self.tool.set_sentence(s)
        num = self.tool.unique_char_num()

        self.hidden_size = num
        self.number_of_class = num # 25
        self.length_of_sequence = len_of_seq  # Any arbitrary number-> 앞의 예제에서는 x_data, ydata 문자 수(전체 문장 - 1)
        repeat_time = len(s) - self.length_of_sequence  # 전체길이: 180, seq_length: 10
        self.batch_size = repeat_time

    def learn(self, dX, dY, total_loop, check_step):

        tf.set_random_seed(777)  # reproducibility

        self.init_network()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('\nStart learning:')

        for i in range(total_loop + 1):
            self.sess.run(self.optimizer, feed_dict={self.X: dX, self.Y: dY})
            l = self.sess.run(self.cost_function, feed_dict={self.X: dX, self.Y: dY})
            predictedY = self.sess.run(self.hypothesis, feed_dict={self.X: dX})
            # print(predictedY.shape) # (171, 10, 65)
            for j, result in enumerate(predictedY):
                index = np.argmax(result, axis=1)  # 가장 안쪽에 있는 리스트에서 가장 큰 값을 갖는 인덱스 반
                # print(i, j, ''.join([self.unique_char_list[t] for t in index]), l)

            self.errors.append(l)

            if i % check_step == 0:
                print_dot()

        print('\nDone.')

    def predict(self, dX):
        # 부분으로 잘라낸 것 여러 개가 들어있는 dataX. 여기에 있는 각 부분을 입력으로 주어 맨 마지막에 예측하는 문자 하나를
        # 을 이용하여
        # Let's print the last char of each result to check it works
        print('\n')
        predictedY = self.sess.run(self.hypothesis, feed_dict={self.X: dX})
        # print(predictedY.shape) # 171, 10, 65

        for j, result in enumerate(predictedY):  # result에는 65개 짜리가 10개 들어있다.
            # 가장 안쪽(axis=1)에 있는 65개 중 가장 큰 값을 갖는 인덱스(0~64중 하나) 출력
            index = np.argmax(result, axis=1)
            # print(j, index)
            if j is 0:  # 맨 처음에는 모든 것을 출력
                s = self.tool.index_to_string(index)
                print(''.join(s), end='')
            else: # 나머지의 경우에는 10개 중 가장 마지막 문자(예측한 문자)만 출력
                s = self.tool.index_to_string([index[-1]])
                print(''.join(s), end='')

    def sentence_to_data(self, sen):
        dX, dY = self.cut_and_append(sen)
        return dX, dY

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

            x_index_list_partial = self.tool.string_to_index(x_sentence_partial)  # x str to index
            y_index_list_partial = self.tool.string_to_index(y_sentence_partial)  # y str to index

            dataX.append(x_index_list_partial)
            dataY.append(y_index_list_partial)

        return dataX, dataY

    def show_error(self):
        from lib.myplot import MyPlot
        mp = MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.errors)

