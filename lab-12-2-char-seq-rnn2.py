# Lab 12 Character Sequence RNN
import tensorflow as tf
import numpy as np

# under refactoring...

class XXX:
    def run(self):
        tf.set_random_seed(777)  # reproducibility

        sample_sentence = " if you want you"
        unique_char_collec = set(sample_sentence)  # set class는 중복된 문자(space 3개, y, o, u)를 제거한 후 무작위로 collection 생성
        # tmp = {'n', 't', 'y', 'w', ' ', 'f', 'u', 'a', 'i', 'o'}

        unique_char_list = list(unique_char_collec)  # index -> char
        # uique_char_list = ['n', 't', 'y', 'w', ' ', 'f', 'u', 'a', 'i', 'o']

        aa = enumerate(unique_char_list)

        # {'y': 0, 'a': 1, 'f': 2, 'o': 3, 'i': 4, 'w': 5, 't': 6, 'u': 7, 'n': 8, ' ': 9}
        unique_char_and_index = {c: i for i, c in aa}
        #print(unique_char_and_index)

        # hyper parameters
        rnn_hidden_size = len(unique_char_and_index)  # RNN output size
        num_classes = len(unique_char_and_index)  # final output size (RNN or softmax, etc.)
        print(num_classes)
        batch_size = 1  # one sample data, one batch
        sequence_length = len(sample_sentence) - 1  # 16 - 1 = 15, number of lstm rollings (unit #)

        # 샘플 문장에 있는 문자 순서대로 인덱스를 구함
        # ' if you want you' 문장 전체에 있는 문자 인덱스 리스트
        char_index_list = [unique_char_and_index[c] for c in sample_sentence]  # char to index

        x_data = [char_index_list[:-1]]  # 가장 끝 문자를 제외한 나머지 문자들의 인덱스 ' if you want yo'의 인덱스 리스트
        print(x_data)
        y_data = [char_index_list[1:]]   # 처음 문자를 제외한 나머지 문자들의 인덱스 'if you want you'의 인덱스 리스트
        print(y_data)

        X = tf.placeholder(tf.int32, [None, sequence_length])  # 15, X data
        Y = tf.placeholder(tf.int32, [None, sequence_length])  # 15, Y label

        # ????? X: [[9, 1, 7, 9, 2, 3, 8, 9, 5, 4, 6, 0, 9, 2, 3]], num_classes: 10
        x_one_hot = tf.one_hot(X, num_classes)  # X: 1 -> x_one_hot: 0 1 0 0 0 0 0 0 0 0

        oh = tf.one_hot([[9, 1, 7, 9, 2, 3, 8, 9, 5, 4, 6, 0, 9, 2, 3]], 10)
        '''
        ' if you  want yo'
        [[[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.] ' '
          [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.] i 
          [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.] f
          [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.] ' '
          [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.] y
          [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.] o 
          [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.] u 
          [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.] ' '
          [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.] w 
          [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.] a
          [ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.] n 
          [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.] t
          [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.] ' '
          [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.] y
          [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]]] o
        '''

        cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs, _states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state, dtype=tf.float32)

        weights = tf.ones([batch_size, sequence_length]) #shape = (1, 15)

        sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
        loss = tf.reduce_mean(sequence_loss)
        train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

        prediction = tf.argmax(outputs, axis=2)

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        for i in range(10): #3000
            l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
            #result = sess.run(prediction, feed_dict={X: x_data})

            # print char using dic
            #result_str = [unique_char_list[c] for c in np.squeeze(result)]

            print(i, "loss:", l) #, "Prediction:", ''.join(result_str))


gildong = XXX()
gildong.run()



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
