# http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
# http://learningtensorflow.com/index.html
# http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint


class XXX:
    X = None
    Y = None

    sess = None

    hypothesis = None

    '''
    def rnn_layer_ph(self, hidden_size):
        # One cell RNN input_dim (4) -> output_dim (2)
        cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
        output, _states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
        return output
    '''
    def rnn_layer_with_RNN_cell(self, xdata, hidden_size):
        # One cell RNN input_dim (4) -> output_dim (2)
        cell = rnn.BasicRNNCell(num_units=hidden_size)
        output, _states = tf.nn.dynamic_rnn(cell, xdata, dtype=tf.float32)

        return output

    def rnn_layer_with_LSTM_cell(self, xdata, hidden_size, seq_len=None, init_state=0):

        if seq_len==None and init_state==0:
            print('seq_len==None and init_state==0')
            cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
            output, _states = tf.nn.dynamic_rnn(cell, xdata, dtype=tf.float32)
        elif seq_len!=None and init_state==0:
            print('seq_len!=None and init_state==0')
            cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
            output, _states = tf.nn.dynamic_rnn(cell, xdata, sequence_length=seq_len, dtype=tf.float32)
        elif seq_len==None and init_state==1:
            print('seq_len==None and init_state==1')
            batch_size = 3
            cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
            initial_state = cell.zero_state(batch_size, tf.float32)
            output, _states = tf.nn.dynamic_rnn(cell, xdata, initial_state=initial_state, dtype=tf.float32)
        return output

        '''
        cell = rnn.BasicRNNCell(num_units=hidden_size)
        outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
        
        cell = rnn.BasicRNNCell(num_units=hidden_size)
        outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
        
        ===========
        cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
        
        cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        outputs, _states = tf.nn.dynamic_rnn(cell, x_data, sequence_length=[5, 3, 4], dtype=tf.float32)
    
        cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, _states = tf.nn.dynamic_rnn(cell, x_data, initial_state=initial_state, dtype=tf.float32)
            
        cell = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, _states = tf.nn.dynamic_rnn(cell, x_data, initial_state=initial_state, dtype=tf.float32)    

        '''

    '''
    def test_ph(self, achar):
        x_data = np.array([[achar]], dtype=np.float32)  # x_data = [[[1,0,0,0]]], shape:3,
        # 데이터(h)를 입력으로 주고 RNN 출력을 구했더니...
        output_list = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        print(output_list)
    '''

    def test(self):
        # 데이터(h)를 입력으로 주고 RNN 출력을 구했더니...
        output_list = self.sess.run(self.hypothesis)
        print('입력에 대한 RNN 출력:')
        print(output_list, '\n')


    def set_hypothesis(self, hy):
        self.hypothesis = hy

    def set_placeholders(self, input_size):
        self.X = tf.placeholder(tf.float32, [None, 1, input_size])
    '''
    def init_network(self):
        self.set_placeholders(4)

        output = self.rnn_layer_ph(hidden_size=2)
        self.set_hypothesis(output)
    '''
    def run(self):

        # One hot encoding for each char in 'hello'
        h = [1, 0, 0, 0]
        e = [0, 1, 0, 0]
        l = [0, 0, 1, 0]
        o = [0, 0, 0, 1]

        #self.init_network()

        self.sess = tf.InteractiveSession()

        # 문자 하나 입력 -> cell
        with tf.variable_scope('one_cell') as scope:
            x_data = np.array([[h]], dtype=np.float32)
            print(x_data.shape)

            outputs = self.rnn_layer_with_RNN_cell(x_data, 2) # shape=(1, 1, 4)
            self.set_hypothesis(outputs)

            self.sess.run(tf.global_variables_initializer())

            #self.test()

        # 문자 5개 입력 -> sequences
        with tf.variable_scope('sequances') as scope:
            # One cell RNN input_dim (4) -> output_dim (2). sequence: 5
            x_data = np.array([[h, e, l, o]], dtype=np.float32)
            print(x_data.shape)

            outputs = self.rnn_layer_with_RNN_cell(x_data, 2) #shape=(1,4,4)
            self.set_hypothesis(outputs)

            self.sess.run(tf.global_variables_initializer())

            #self.test()

        # 문자 5개짜리를 3번 입력 : batches
        with tf.variable_scope('batches') as scope:
            # One cell RNN input_dim (4) -> output_dim (2). sequence: 5, batch 3
            # 3 batches 'hello', 'eolll', 'lleel'
            x_data = np.array([[h, e, l, l, o],
                               [e, o, l, l, l],
                               [l, l, e, e, l]], dtype=np.float32)
            print(x_data.shape)

            outputs = self.rnn_layer_with_LSTM_cell(x_data, 2) # shape=(3,5,4)
            self.set_hypothesis(outputs)

            self.sess.run(tf.global_variables_initializer())

            #self.test()

        # 문자 5개짜리 각각에 대하여 sequence_length를 지정
        with tf.variable_scope('batches_dynamic_length') as scope:
            # One cell RNN input_dim (4) -> output_dim (5). sequence: 5, batch 3
            # 3 batches 'hello', 'eolll', 'lleel'
            x_data = np.array([[h, e, l, l, o],
                               [e, o, l, l, l],
                               [l, l, e, e, l]], dtype=np.float32) #shape=(3,5,4)
            print('seq_len', x_data.shape)

            output = self.rnn_layer_with_LSTM_cell(x_data, 2, [5, 3, 4])
            self.set_hypothesis(output)

            self.sess.run(tf.global_variables_initializer())

            #self.test()


        with tf.variable_scope('initial_state') as scope:
            #batch_size = 3
            x_data = np.array([[h, e, l, l, o],
                               [e, o, l, l, l],
                               [l, l, e, e, l]], dtype=np.float32)
            print('initial')

            self.rnn_layer_with_LSTM_cell(x_data, 2, None, 1)

            # One cell RNN input_dim (4) -> output_dim (5). sequence: 5, batch: 3
            #hidden_size = 2
            #cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True)
            #initial_state = cell.zero_state(batch_size, tf.float32)
            #outputs, _states = tf.nn.dynamic_rnn(cell, x_data, initial_state=initial_state, dtype=tf.float32)

            self.sess.run(tf.global_variables_initializer())
            #self.test()
            #print(outputs.eval())


        # 생성된 데이터(3,5,3), 3차원 입력->5시퀀스->3배치, hidden_size=5이므로 5차원 출력
        batch_size=3
        sequence_length=5
        input_dim=3

        x_data = np.arange(45, dtype=np.float32).reshape(batch_size, sequence_length, input_dim)
        #print(x_data)  # batch, sequence_length, input_dim

        with tf.variable_scope('generated_data') as scope:
            # One cell RNN input_dim (3) -> output_dim (5). sequence: 5, batch: 3

            output = self.rnn_layer_with_LSTM_cell(x_data, 5, None, 1)
            self.set_hypothesis(output)

            #cell = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
            #initial_state = cell.zero_state(batch_size, tf.float32)
            #outputs, _states = tf.nn.dynamic_rnn(cell, x_data, initial_state=initial_state, dtype=tf.float32)
            self.sess.run(tf.global_variables_initializer())

            # 이러한 입력을 넣을 경우(3X5X3) 나오는 출력
            #self.test()


        # 다층 RNN : 5개 히든 뉴런이 3층으로 연결된 RNN
        with tf.variable_scope('MultiRNNCell') as scope:
            # Make rnn
            cell = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True) # 5=히든=출력
            cell = rnn.MultiRNNCell([cell] * 3, state_is_tuple=True) # 5 히든이 3 layers

            # rnn in & out
            # 3차원 입력, 5차원 출력 -> 5시퀀스 & 3배치
            # cell 네트워크에 x_data 입력을 줄 때 출력이 outputs임.
            outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

            print("dynamic rnn outputs 텐서: ", outputs)
            self.sess.run(tf.global_variables_initializer())

            # 다층 RNN 출력을 구하라.
            #print(outputs.eval())  # batch size, unrolling (time), hidden_size

        # 5개의(5차원) 출력 뉴런을 갖는 RNN, 3-5-3 입력 -> 3 * 5개의 5차원 출력
        # 이때 각 배치별 출력 수 지정 [1, 3, 2]
        with tf.variable_scope('dynamic_rnn') as scope:

            #print(x_data)
            cell = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True) # 출력 5
            outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32, sequence_length=[1, 3, 2])
            # lentgh 1 for batch 1, lentgh 3 for batch 2, length 2 for batch 3

            print("dynamic rnn의 outputs 텐서 ", outputs)
            self.sess.run(tf.global_variables_initializer())
            #print(outputs.eval())  # batch size, unrolling (time), hidden_size

        # 5차원 출력을 갖는 cell 2개. 이를 이용하여 양방향 RNN을 구성하고 입력 x_data를 지정. 배치별 시퀀스 길이 지정.
        # 이때 각 셀별로 출력값이 표시됨.
        with tf.variable_scope('bi-directional') as scope:
            # bi-directional rnn
            cell_fw = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)
            cell_bw = rnn.BasicLSTMCell(num_units=5, state_is_tuple=True)

            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x_data,
                sequence_length=[2, 3, 1], dtype=tf.float32)

            self.sess.run(tf.global_variables_initializer())

            #print(self.sess.run(outputs))
            # states 출력 결과는 아리송
            #print(self.sess.run(states))


        # flattern based softmax
        hidden_size=3
        sequence_length=5
        batch_size=3
        num_classes=5

        #print(x_data) # hidden_size=3, sequence_length=4, batch_size=2
        x_data = x_data.reshape(-1, hidden_size)
        #print(x_data)

        softmax_w = np.arange(15, dtype=np.float32).reshape(hidden_size, num_classes) #3, 5
        #print(softmax_w)
        outputs = np.matmul(x_data, softmax_w)
        outputs = outputs.reshape(-1, sequence_length, num_classes) # batch, seq, class
        #print(outputs)

        # 의미있는 코드!!!
        # [batch_size, sequence_length]
        y_data = tf.constant([[1, 1, 1]])  #(1, 3)

        # [batch_size, sequence_length, emb_dim ]
        prediction = tf.constant([[[0, 1], [1, 0], [0, 1]]], dtype=tf.float32) #(1, 3, 2)

        # [batch_size * sequence_length]
        weights = tf.constant([[1, 1, 1]], dtype=tf.float32) #(1, 3)

        # 시스템의 결과, 실제 맞춰야 할 값, 조정(학습)해야 할 가중치
        # 위 3가지를 주고 오류를 구하라.
        sequence_loss = tf.contrib.seq2seq.sequence_loss(prediction, y_data, weights)
        self.sess.run(tf.global_variables_initializer())
        print("Loss: ", sequence_loss.eval())


        # [batch_size, sequence_length]
        y_data = tf.constant([[1, 1, 1]])

        # [batch_size, sequence_length, emb_dim ]
        prediction1 = tf.constant([[[0, 1], [0, 1], [0, 1]]], dtype=tf.float32)
        prediction2 = tf.constant([[[1, 0], [1, 0], [1, 0]]], dtype=tf.float32)
        prediction3 = tf.constant([[[0, 1], [1, 0], [0, 1]]], dtype=tf.float32)

        # [batch_size * sequence_length]
        weights = tf.constant([[1, 1, 1]], dtype=tf.float32)

        sequence_loss1 = tf.contrib.seq2seq.sequence_loss(prediction1, y_data, weights)
        sequence_loss2 = tf.contrib.seq2seq.sequence_loss(prediction2, y_data, weights)
        sequence_loss3 = tf.contrib.seq2seq.sequence_loss(prediction3, y_data, weights)

        self.sess.run(tf.global_variables_initializer())
        print("Loss1: ", sequence_loss1.eval(),
              "Loss2: ", sequence_loss2.eval(),
              "Loss3: ", sequence_loss3.eval())


gildong = XXX()
gildong.run()



