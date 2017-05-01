import tensorflow as tf
import matplotlib.pyplot as plt
from lib.common_db import CommonDB
from abc import abstractmethod


class StockRNN:

    input_dim = 0 #5 input size?
    output_dim = 0 #3  단순 rnn일 때는 1
    seq_length = 0 #7 time steps

    X = None
    Y = None

    hypothesis = None
    cost_function = None
    optimizer = None

    errors = []

    @abstractmethod
    def init_network(self):
        pass

    def set_parameter(self, input_dim, seq_length, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_length = seq_length

    def set_placeholder(self, seq_len, d_dim):
        # input place holders
        self.X = tf.placeholder(tf.float32, [None, seq_len, d_dim])
        self.Y = tf.placeholder(tf.float32, [None, 1])

    def create_simple_rnn_layer(self, output_dim):
        # create_rnn_layer(output_dim)
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.output_dim, state_is_tuple=True)
        outputs, _states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32) #X shape=(?, 7, 5)
        #print(outputs)
        pred = outputs[:, -1]  # 1개짜리 7개 중 가장 마지막 1개짜리를 출력으로 선택함.
        return pred

    def create_multi_rnn_softmax_layer(self):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.output_dim, state_is_tuple=True)  #output_dim = 3
        cell = tf.contrib.rnn.MultiRNNCell([cell] * 2, state_is_tuple=True)
        outputs, _states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32) #shape of X = (?, 7, 5)
        #print('output', outputs) # 위 output_dim이 3일 때 3차원 출력이 seq_length(7)만큼 나옴
        last_output = outputs[:, -1]  # 3차원 출력 7개 중 가장 마지막 3차원 출력을 최종 출력으로 선택함. last_output shape=(?, 3)

        # Softmax layer (rnn_hidden_size -> num_classes)
        W = tf.get_variable("softmax_w", [self.output_dim, 1])
        b = tf.get_variable("softmax_b", [1])
        Y_pred = tf.matmul(last_output,  W) + b  # 3 *
        #print(Y_pred, last_output, softmax_w)

        return Y_pred # (?, 1).. 결국 (1, 7, 5) 데이터가 들어가면 (1, 1) 데이터가 출력됨.

    def set_hypothesis(self, hypo):
        self.hypothesis = hypo

    def set_cost_function(self):
        # cost/loss
        self.cost_function = tf.reduce_sum(tf.square(self.hypothesis - self.Y))  # sum of the squares

    def set_optimizer(self, l_rate):
        # optimizer
        optimizer = tf.train.GradientDescentOptimizer(l_rate)
        self.optimizer = optimizer.minimize(self.cost_function)

    def learn(self, trainX, trainY, total_loop, check_step):
        tf.set_random_seed(777)  # reproducibility

        self.init_network()
        if self.input_dim == 0 | self.output_dim == 0 | self.seq_length == 0:
            print('Set RNN parameters!')
            exit()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('\nStart learning:')

        for i in range(total_loop):
            self.sess.run(self.optimizer, feed_dict={self.X: trainX, self.Y: trainY})
            loss = self.sess.run(self.cost_function, feed_dict={self.X: trainX, self.Y: trainY})
            self.errors.append(loss)

            if i % check_step == 0:
                from lib import mytool
                mytool.print_dot()

        print('\nDone!\n')

    def predict(self, testX, testY):
        # RMSE
        predY = self.sess.run(self.hypothesis, feed_dict={self.X: testX})
        rmse = tf.sqrt(tf.reduce_mean(tf.square(testY - predY))) # 차의 제곱의 평균의 sqrt
        print("RMSE", self.sess.run(rmse))

        plt.plot(testY)  # 실제
        plt.plot(predY)  # 예측
        plt.xlabel("Time Period")
        plt.ylabel("Stock Price")
        plt.show()

    def show_error(self):
        from lib.myplot import MyPlot
        mp = MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.errors)
