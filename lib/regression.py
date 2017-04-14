from abc import abstractmethod

import myplot
import mytool
import tensorflow as tf
from file2buffer import File2Buffer

from lib.nntype import NNType


class Regression:
    # placeholder
    X = None
    Y = None

    # weight & bias
    W = None
    b = None

    # hypothesis, cost funcion, optimizer
    hypothesis = None
    cost_function = None
    optimizer = None

    costs = []
    weights = []
    biases = []
    logs = []

    sess = None

    #[1]
    def set_placeholder(self, x_dim, y_dim):
        self.X = tf.placeholder(tf.float32, shape=[None, x_dim])
        self.Y = tf.placeholder(tf.float32, shape=[None, y_dim])
        '''
        self.X = tf.placeholder("float")
        self.Y = tf.placeholder("float")
        self.X = tf.placeholder(tf.float32, shape=[None])
        self.Y = tf.placeholder(tf.float32, shape=[None])
        '''

    #[2]
    def set_weight_bias(self, x_dim, y_dim):
        self.W = tf.Variable(tf.random_normal([x_dim, y_dim]), name='weight')
        self.b = tf.Variable(tf.random_normal([y_dim]), name='bias')
        '''
        self.W = tf.Variable(numpy.random.randn(), "weight")
        self.b = tf.Variable(numpy.random.randn(), "bias")
        self.W = tf.Variable(tf.random_normal([1]), name='weight')
        self.b = tf.Variable(tf.random_normal([1]), name='bias')
        self.W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 리스트로 리턴
        self.b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 리스트로 리턴
        '''

    #[3]
    def set_hypothesis(self, type):
        if type == NNType.SQUARE_MEAN: #linear regression
            self.hypothesis = tf.add(tf.matmul(self.X, self.W), self.b)
        elif type == NNType.LOGISTIC: #logistic regression
            # Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
            self.hypothesis = tf.sigmoid(tf.add(tf.matmul(self.X, self.W), self.b))
        elif type == NNType.SOFTMAX: #softmax
            # tf.nn.softmax computes softmax activations
            # softmax = exp(logits) / reduce_sum(exp(logits), dim)
            self.hypothesis = tf.nn.softmax(tf.add(tf.matmul(self.X, self.W), self.b))
        elif type == 4:
            # tf.nn.softmax computes softmax activations
            # softmax = exp(logits) / reduce_sum(exp(logits), dim)
            logits = tf.matmul(self.X, self.W) + self.b
            self.hypothesis = tf.nn.softmax(logits)

        '''
        self.hypothesis = tf.add(tf.mul(self.X, self.W), self.b)  # W * x_data + b
        self.hypothesis = self.X * self.W + self.b
        '''

    #[4]
    def set_cost_function(self, type):
        if type == NNType.SQUARE_MEAN: #linear
            self.cost_function = tf.reduce_mean(tf.square(self.hypothesis - self.Y))
        elif type == NNType.LOGISTIC: #logistic
            self.cost_function = -tf.reduce_mean(self.Y * tf.log(self.hypothesis) + (1 - self.Y) * tf.log(1 - self.hypothesis))
        elif type == NNType.SOFTMAX: #softmax
            # Cross entropy cost/loss
            self.cost_function = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.hypothesis), axis=1))


    #[5] l_rate = 0.1 or 0.01 or 1e-5
    def set_optimizer(self, l_rate):
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=l_rate).minimize(self.cost_function)

    def show_error(self):
        mp = myplot.MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.costs)

    def print_error(self):
        for item in self.costs:
            print(item)

    def show_weight(self):
        print('shape=', self.weights)

        if len(self.weights[0]) is 1:
            mp = myplot.MyPlot()
            mp.set_labels('Step', 'Weight')
            mp.show_list(self.weights)
        else:
            print('Cannot show the weight! Call print_weight method.')

    def print_weight(self):
        for item in self.weights:
            print(item)

    def show_bias(self):
        if len(self.weights) is 1:
            mp = myplot.MyPlot()
            mp.set_labels('Step', 'Bias')
            mp.show_list(self.biases)
        else:
            print('Cannot show the bias! Call print_bias mehtod.')

    def print_bias(self):
        for item in self.biases:
            print(item)

    def print_log(self):
        for item in self.logs:
            print(item)

    @abstractmethod
    def init_network(self):
        pass

    @abstractmethod
    def my_log(self, i, x_data, y_data):
        err = self.sess.run(self.cost_function, feed_dict={self.X: x_data, self.Y: y_data})
        msg = "Step:{}, Error:{:.6f}".format(i, err)
        self.logs.append(msg)

    #
    def check_step_processing(self, i, x_data, y_data):
        mytool.print_dot()
        self.weights.append(self.sess.run(self.W))
        self.biases.append(self.sess.run(self.b))
        err = self.sess.run(self.cost_function, feed_dict={self.X: x_data, self.Y: y_data})
        self.costs.append(err)
        self.my_log(i, x_data, y_data)  # Do whatever you want in this virtual function

    def learn(self, xdata, ydata, total_loop, check_step):
        tf.set_random_seed(777)  # for reproducibility, call before init method

        self.init_network()

        print('\nStart learning.')

        if self.sess is None:
            # Launch the graph in a session.
            self.sess = tf.Session()

        # Initializes global variables in the graph.
        self.sess.run(tf.global_variables_initializer())

        # 옵티마이저로 학습(W, b를 수정)
        # 지정한 간격으로 W, b를 리스트에 저장하고 그 때의 오류값도 리스트에 저장
        # 원하는 것을 할 수 있도록
        for i in range(total_loop + 1):
            self.sess.run(self.optimizer, feed_dict={self.X: xdata, self.Y: ydata})

            if i % check_step == 0:
                self.check_step_processing(i, xdata, ydata)

        print('\nDone!\n')

    def load_file(self, afile):
        f2b = File2Buffer()
        f2b.file_load(afile)
        return f2b.x_data, f2b.y_data

    def learn_from_file(self, afile, total_loop, check_step):
        f2b = File2Buffer()
        f2b.file_load(afile)
        #f2b.print_info()

        self.learn(f2b.x_data, f2b.y_data, total_loop, check_step)

    # 여러 (대용량) 파일들(파일 리스트)을 주고 학습하도록 시킴
    def learn_batch(self, file_list, total_loop, check_step):
        tf.set_random_seed(777)  # for reproducibility

        filename_queue = tf.train.string_input_producer(file_list, shuffle=False, name='filename_queue')
        reader = tf.TextLineReader()
        key, value = reader.read(filename_queue)

        # Default values, in case of empty columns. Also specifies the type of the
        # decoded result.
        record_defaults = [[0.], [0.], [0.], [0.]]
        xy = tf.decode_csv(value, record_defaults=record_defaults)

        # collect batches of csv in
        train_x_batch, train_y_batch = tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

        self.init_network()

        # Launch the graph in a session.
        self.sess = tf.Session()
        # Initializes global variables in the graph.
        self.sess.run(tf.global_variables_initializer())

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        print('\nStart learning...')

        for step in range(total_loop + 1):
            x_data, y_data = self.sess.run([train_x_batch, train_y_batch])
            err_val = self.sess.run(self.optimizer, feed_dict={self.X: x_data, self.Y: y_data})

            if step % check_step == 0: #10
                self.check_step_processing(step, x_data, y_data)

        print('\nDone!\n')

        coord.request_stop()
        coord.join(threads)

    #predict, recognize, classify
    def test(self, x_data):
        # Test our model
        for item in x_data:
            print(item)
        print('->')
        answer = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        for item in answer:
            print(item)
        print('\n')

    def test_argmax(self, x_data):
        # Testing & One-hot encoding
        for item in x_data:
            print(item)
        print('->')
        answer = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        for item in answer:
            print(item)

        print(self.sess.run(tf.arg_max(answer, 1)))
        print('\n')

    def evaluate(self, x_data, y_data):
        # Accuracy computation
        # True if hypothesis>0.5 else False
        predicted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, self.Y), dtype=tf.float32))

        # Accuracy report
        h, c, a = self.sess.run([self.hypothesis, predicted, accuracy], feed_dict={self.X: x_data, self.Y: y_data})
        print("\nHypothesis: ", h, "\nCorrect: ", c, "\nAccuracy: ", a)

    def recognition_rate(self, x_data, y_data):
        correct_prediction = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y_one_hot, 1)) # !!! Y_one_hot은 파생 클래스에 있는 것임!!
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = self.sess.run(accuracy, feed_dict={self.X: x_data, self.Y: y_data})
        print("{:.2%}".format(acc))

    '''
    def recognition_rate(self, x_data, y_data):
        # Accuracy computation. True if hypothesis>0.5 else False
        answer = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(answer, self.Y), dtype=tf.float32))

        h, ans, a = self.sess.run([self.hypothesis, answer, accuracy], feed_dict={self.X: x_data, self.Y: y_data})
        str = '\nHypothesis:{}, \nAnswer (Y):{}, \nAccuracy:{}'.format(h, ans, a)
        self.logs.append(str)
        self.print_log()
    '''
    '''
    # Accuracy computation
    # True if hypothesis>0.5 else False
    predicted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, self.Y), dtype=tf.float32))

    # Accuracy report
    h, c, a = self.sess.run([self.hypothesis, predicted, accuracy], feed_dict={self.X: self.x_data, self.Y: self.y_data})
    print(self.sess.run(self.hypothesis, feed_dict={self.X: [[0.176471,0.155779,0,0,0,0.052161,-0.952178,-0.733333]]}))

    #print("\n예측한 값: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

    '''


'''
--- sigle variable
x_data = [1]
y_data = [1]

--- multi-variables
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

--- Single variable
x_data = [1, 2, 3]
y_data = [1, 2, 3]
gildong.learn(x_data, y_data, 2000, 50)
gildong.test([5, 2.5, 1.5, 3.5])
gildong.learn([1, 2, 3, 4, 5], [2.1, 3.1, 4.1, 5.1, 6.1], 2000, 50)
gildong.test([5, 2.5, 1.5, 3.5])
'''
