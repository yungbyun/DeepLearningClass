import tensorflow as tf
from abc import abstractmethod
import lib.mytool as mytool


class RNNCore3:
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
    batch_size = 1

    @abstractmethod
    def init_network(self):
        pass

    @abstractmethod
    def create_writer(self): # for tensorboard
        pass

    @abstractmethod
    def do_summary(self, feed_dict): # for tensorboard
        pass

    def set_parameters(self, unique_char_num, seq_num):
        self.input_size = unique_char_num # 유일한 문자 수
        self.hidden_size = unique_char_num # 유일한 문자 수
        self.output_size = unique_char_num # 유일한 문자 수
        self.number_of_class = unique_char_num # 유일한 문자 수
        self.length_of_sequence = seq_num # x_data, y_data 문자 수

    def set_placeholder(self, seq_len, hidden_size):
        self.X = tf.placeholder(tf.float32, [None, seq_len, hidden_size])  # None, 12, 10
        self.Y = tf.placeholder(tf.float32, [None, seq_len, hidden_size])  # None, 12, 10

    # 아래 함수는 hypothesis를 정의하므로 learn에서 cost_function, optimizer, predict를 실행하면 결국 이 hypothesis가 실행됨.
    def rnn_lstm_cell(self, X, num_classes, hidden_size, batch_size):
        # X: [[9, 1, 7, 9, 2, 3, 8, 9, 5, 4, 6, 0, 9, 2, 3]], num_classes: 10
        # X로 입력받는 숫자 각각에 대하여 num_classes 개의 0 중 해당 위치만 1로 만드는 텐서를 리턴함.
        #x_one_hot = tf.one_hot(X, num_classes)  # X: 1 -> x_one_hot: 0 1 0 0 0 0 0 0 0 0
        #print(x_one_hot) #(1, 15, 10), (1, 6, 5), (1, 12, 10)

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)  # 10
        initial_state = cell.zero_state(batch_size, tf.float32)  # 1
        hypothesis, _states = tf.nn.dynamic_rnn(cell, self.X, initial_state=initial_state, dtype=tf.float32)
        print(hypothesis)
        # shape = (1, 15, 10) 글자 하나를 의미하는 10차원 출력 벡터 -> 15개 출력됨.
        # shape = (1, 6, 5) 'hihello'
        # shape = (1, 12, 10) ' hello,world!'
        return hypothesis

    def set_hypothesis(self, hypo):
        self.hypothesis = hypo

    def set_cost_function(self, batch_size, seq_len):
        weights = tf.ones([batch_size, seq_len])
        self.cost_function = tf.contrib.seq2seq.sequence_loss(logits=self.hypothesis, targets=self.Y, weights=weights)
        # sequence_loss = tf.nn.seq2seq.sequence_loss_by_example(logits=outputs, targets=Y, weights=weights)

    def set_optimizer(self, l_rate):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(tf.reduce_mean(self.cost_function))

    def print_log(self):
        for item in self.logs:
            print(item)

    def learn(self, x_index_list, y_index_list, total_loop, check_step):
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

        print('error', self.sess.run(self.cost_function, feed_dict={self.X: [x_index_list], self.Y: [y_index_list]}))

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

    def predict(self, x_index_list):
        print('\nPrediction:')

        prediction = tf.argmax(self.hypothesis, axis=2)
        print('hypo', self.hypothesis, 'pred', prediction)

        #print(self.sess.run(self.hypothesis, feed_dict={self.X: [x_index_list]}))
        '''  문자열이 ' hello,world!'일때 (전체 문자 수 = 13, 유일한 문자의 수 = 10, 시퀀스 크기 = 12 
        [[[ -7.60889530e-01  -7.61245489e-01   7.61426389e-01  -7.60769010e-01  2.92712706e-04  -7.60979712e-01  -7.60919273e-01  -7.60645986e-01  -7.61091411e-01   2.24163537e-04]
          [ -9.63875651e-01  -9.63916421e-01  -7.61444509e-01  -9.63519752e-01  9.63881433e-01   2.09505345e-07  -9.63651180e-01  -9.63523090e-01  -9.63534832e-01   3.08772025e-04]
          [ -9.95005786e-01  -9.95038867e-01  -9.63946640e-01  -9.94904399e-01 -7.60594845e-01   9.02319330e-11  -9.94967639e-01  -9.94871318e-01  -9.94430423e-01   9.94974613e-01]
          [ -5.35215557e-01  -9.99327421e-01  -9.95024443e-01  -9.99302208e-01 -9.63744700e-01   1.18589848e-02  -9.98280883e-01  -9.99160469e-01  -9.98650908e-01   9.96824741e-01]
          [  6.11776626e-03  -9.99908924e-01  -9.99323547e-01  -9.99903798e-01 -9.95005310e-01   9.96469498e-01  -9.99659717e-01  -9.99661803e-01  -9.99683619e-01  -7.36088276e-01]
          [  9.60394263e-01  -9.99987423e-01  -9.99908090e-01  -9.99291837e-01 -9.99211133e-01  -7.61046708e-01  -9.99948144e-01   2.61888683e-01  -9.99956906e-01  -9.58742678e-01]
          [ -7.60107040e-01  -9.99974608e-01  -9.99959528e-01  -9.94525969e-01 -9.99845624e-01   2.54319757e-01  -9.99961793e-01   8.53177905e-01  -9.99958813e-01  -9.92042959e-01]
          [ -9.63886976e-01  -9.99993503e-01  -9.99993265e-01   1.98726416e-01 -9.99944687e-01   8.50999951e-01  -9.99993205e-01  -7.60818124e-01  -9.99987900e-01  -9.98903275e-01]
          [ -9.94598150e-01  -9.99999583e-01  -9.99999464e-01   8.33957970e-01 -9.99989092e-01  -7.60911524e-01  -9.99997735e-01  -9.61866200e-01  -9.99991894e-01  -9.99726295e-01]
          [ -9.99267161e-01  -9.99999642e-01  -9.99999166e-01  -7.60838807e-01 -9.99989212e-01  -9.63607967e-01   2.22954378e-01  -9.94749427e-01  -9.99761939e-01   7.61457145e-01]
          [ -9.98854220e-01  -9.99998808e-01  -9.99999642e-01  -9.63535964e-01 -9.99997735e-01  -9.92970109e-01   8.41399848e-01  -9.99194026e-01   1.79880947e-01  -7.59848118e-01]
          [ -9.99900341e-01  -9.99979973e-01  -9.99982417e-01  -9.94852543e-01 -9.99965668e-01  -9.99042451e-01  -7.60782719e-01  -9.99876380e-01   8.27998161e-01  -9.63666797e-01]]]
         '''
        predicted = self.sess.run(prediction, feed_dict={self.X: [x_index_list]})
        return predicted

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

