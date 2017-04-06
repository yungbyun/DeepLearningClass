import tensorflow as tf
from myplot import MyPlot
from abc import abstractmethod
from nntype import NNType
import mytool
from file2buffer import File2Buffer

class NeuralNetwork:
    # place holders
    X = None
    Y = None
    DO = None #place holer for dropout

    hypothesis = None
    cost_function = None
    optimizer = None

    sess = None

    costs = []
    weights = []
    biases = []
    logs = []

    initializer = None # for weights

    class_num_for_onehot = None

    def onehot(self, class_num):
        self.class_num_for_onehot = class_num

    @abstractmethod
    def init_network(self):
        pass

    @abstractmethod
    def my_log(self, i, xdata, ydata):
        err = self.sess.run(self.cost_function, feed_dict={self.X: xdata, self.Y: ydata})
        msg = "Step:{}, Error:{:.6f}".format(i, err)
        self.logs.append(msg)

    @abstractmethod
    def set_weight_initializer(self):
        pass

    def set_placeholder(self, num_of_input, num_of_output):
        self.X = tf.placeholder(tf.float32, [None, num_of_input])
        self.Y = tf.placeholder(tf.float32, [None, num_of_output])

        '''
        self.X = tf.placeholder("float")
        self.Y = tf.placeholder("float")
        self.X = tf.placeholder(tf.float32, shape=[None])
        self.Y = tf.placeholder(tf.float32, shape=[None])
        '''
    '''
    def set_weight_bias(self, x_dim, y_dim):
        self.W = tf.Variable(tf.random_normal([x_dim, y_dim]), name='weight')
        self.b = tf.Variable(tf.random_normal([y_dim]), name='bias')

        #self.W = tf.Variable(numpy.random.randn(), "weight")
        #self.b = tf.Variable(numpy.random.randn(), "bias")
        #self.W = tf.Variable(tf.random_normal([1]), name='weight')
        #self.b = tf.Variable(tf.random_normal([1]), name='bias')
        #self.W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 리스트로 리턴
        #self.b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 리스트로 리턴
     '''

    def create_layer(self, previous_output, num_of_input, num_of_neuron, w_name, b_name):

        self.set_weight_initializer() ## a hole for you to set an initializer

        if self.initializer == NNType.XAIVER:
            # http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
            W = tf.get_variable(w_name, shape=[num_of_input, num_of_neuron], initializer = tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.random_normal([num_of_neuron]), name = b_name)
        else : # if self.initializer == None:
            W = tf.Variable(tf.random_normal([num_of_input, num_of_neuron]), name = w_name)
            b = tf.Variable(tf.random_normal([num_of_neuron]), name = b_name)

        #output = self.get_neuron_output(previous_output, hypothesis_type, W, b)
        output = tf.add(tf.matmul(previous_output, W), b)
        return output

    def get_neuron_output(self, previous_output, hypothesis_type, W, b):
        output = None
        if previous_output is None: # if it is input layer
            if hypothesis_type == NNType.SQUARE_MEAN:
                output = tf.add(tf.matmul(self.X, W), b)
            elif hypothesis_type == NNType.LOGISTIC:
                output = tf.sigmoid(tf.add(tf.matmul(self.X, W), b))
            elif hypothesis_type == NNType.SOFTMAX:  # softmax
                # tf.nn.softmax computes softmax activations
                # softmax = exp(logits) / reduce_sum(exp(logits), dim)
                logits = tf.add(tf.matmul(self.X, W), b)
                output = tf.nn.softmax(logits)
            elif hypothesis_type == NNType.SOFTMAX_LOGITS:
                # tf.nn.softmax computes softmax activations
                # softmax = exp(logits) / reduce_sum(exp(logits), dim)
                logits = tf.matmul(self.X, W) + b
                output = tf.nn.softmax(logits)
            elif hypothesis_type == NNType.RELU:
                output = tf.nn.relu(tf.add(tf.matmul(self.X, W), b))

        else: # if it is not input layer
            if hypothesis_type == NNType.SQUARE_MEAN:
                output = tf.add(tf.matmul(previous_output, W), b)
            elif hypothesis_type == NNType.LOGISTIC:
                output = tf.sigmoid(tf.matmul(previous_output, W) + b)
            elif hypothesis_type == NNType.SOFTMAX:  # softmax
                logits = tf.add(tf.matmul(previous_output, W), b)
                output = tf.nn.softmax(logits)
            elif hypothesis_type == NNType.SOFTMAX_LOGITS:
                logits = tf.add(tf.matmul(previous_output, W), b)
                output = tf.nn.softmax(logits)
            elif hypothesis_type == NNType.RELU:
                output = tf.nn.relu(tf.matmul(previous_output, W) + b)

        return output

        '''
        self.hypothesis = tf.add(tf.mul(self.X, self.W), self.b)  # W * x_data + b
        self.hypothesis = self.X * self.W + self.b
        '''

    def set_hypothesis(self, h):
        self.hypothesis = h

    def set_cost_function(self, type):
        if type == NNType.SQUARE_MEAN: #linear
           self.cost_function = tf.reduce_mean(tf.square(self.hypothesis - self.Y))
        elif type == NNType.LOGISTIC:
           self.cost_function = -tf.reduce_mean(self.Y * tf.log(self.hypothesis) + (1 - self.Y) * tf.log(1 - self.hypothesis))
        elif type == NNType.SOFTMAX:
           # Cross entropy cost/loss function
           self.cost_function = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.hypothesis), axis=1))
        elif type == NNType.SOFTMAX_LOGITS:
           # logits = tf.matmul(self.X, self.W) + self.b
           # self.hypothesis = tf.nn.softmax(logits)
           # define cost/loss & optimizer
           self.cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.hypothesis, labels=self.Y))

    def set_optimizer(self, type, l_rate):
        if type == NNType.GRADIENT_DESCENT:
           self.optimizer = tf.train.GradientDescentOptimizer(l_rate).minimize(self.cost_function)
        elif type == NNType.ADAM:
           self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self.cost_function)

    def show_error(self):
        mp = MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.costs)

    def print_error(self):
        for item in self.costs:
           print(item)

    def show_weight(self):
        print('shape=', self.weights)

        if len(self.weights[0]) is 1:
           mp = MyPlot()
           mp.set_labels('Step', 'Weight')
           mp.show_list(self.weights)
        else:
           print('Cannot show the weight! Call print_weight method.')

    def print_weight(self):
        for item in self.weights:
           print(item)

    def show_bias(self):
        if len(self.weights) is 1:
           mp = MyPlot()
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

    def check_step_processing(self, i, x_data, y_data):
        mytool.print_dot()
        #self.weights.append(self.sess.run(self.W))
        #self.biases.append(self.sess.run(self.b))
        err = self.sess.run(self.cost_function, feed_dict={self.X: x_data, self.Y: y_data})
        self.costs.append(err)

    def learn(self, xdata, ydata, total_loop, check_step):
        tf.set_random_seed(777)

        self.init_network()  # override

        if self.sess is None:
            # Launch the graph in a session.
            self.sess = tf.Session()

        # Initialize TensorFlow variables
        self.sess.run(tf.global_variables_initializer())

        # 옵티마이저로 학습(W, b를 수정)
        # 지정한 간격으로 W, b를 리스트에 저장하고 그 때의 오류값도 리스트에 저장
        # 원하는 것을 할 수 있도록
        print('\nStart learning:')
        for i in range(total_loop + 1):  # 10,001
            self.sess.run(self.optimizer, feed_dict={self.X: xdata, self.Y: ydata})

            if i % check_step == 0:
                self.check_step_processing(i, xdata, ydata)
                self.my_log(i, xdata, ydata)  # override to do whatever you want with the parameters

        print('\nDone!\n')

    # 마지막이 출력 나머지는 입력인 파일을 읽어옮
    def load_file(self, afile):
        f2b = File2Buffer()
        f2b.file_load(afile)
        return f2b.x_data, f2b.y_data

    # 데이터가 들어있는 파일 하나를 주고 학습하도록 시킴
    def learn_with_file(self, afile, total_loop, check_step):
        f2b = File2Buffer()
        f2b.file_load(afile)
        #f2b.print_info()
        self.learn(f2b.x_data, f2b.y_data, total_loop, check_step)

    # 여러 파일들(파일 리스트)을 주고 학습하도록 시킴
    def learn_with_files(self, file_list, total_loop, check_step):
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

        if self.sess is None:
            # Launch the graph in a session.
            self.sess = tf.Session()

        # Initializes global variables in the graph.
        self.sess.run(tf.global_variables_initializer())

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        print('\nStart learning:')

        for step in range(total_loop + 1):
            x_data, y_data = self.sess.run([train_x_batch, train_y_batch])
            err_val = self.sess.run(self.optimizer, feed_dict={self.X: x_data, self.Y: y_data})

            if step % check_step == 0: #10
                self.check_step_processing(step, x_data, y_data)

        print('\nDone!\n')

        coord.request_stop()
        coord.join(threads)

    # MNIST와 같은 데이터를 이용한 학습
    def learn_with_segment(self, db, learning_epoch, partial_size):
        tf.set_random_seed(777)  # for reproducibility

        self.init_network()  # 가상함수

        self.sess = tf.Session()
        # Initialize TensorFlow variables
        self.sess.run(tf.global_variables_initializer())

        print("\nStart learning:")
        # Training cycle
        for epoch in range(learning_epoch):
            err_4_all_data = 0
            number_of_segment = self.get_number_of_segment()  # 가상함수

            # 처음 데이터를 100개를 읽어 최적화함.
            # 그 다음 100개 데이터에 대하여 수행.
            # 이를 모두 550번 수행하면 전체 데이터 55,000개에 대해 1번 수행하게 됨.
            # 아래 for 문장이 한번 모두 실행되면 전체 데이터에 대해 1번 실행(학습)함.
            for i in range(number_of_segment):
                x_data, y_data = self.get_next_segment()  # 가상함수

                # 아래 에러는 일부분(100개)에 대한 것이므로 전체 에러를 구하려면 550으로 나누어주어야 함. 아래에서 수행
                err_4_partial, _= self.sess.run([self.cost_function, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data})
                err_4_all_data += err_4_partial

            import mytool
            mytool.print_dot()
            avg_err = err_4_all_data / number_of_segment #
            self.costs.append(avg_err)

            self.my_log(epoch, x_data, y_data)  # 가상함수

        print("\nDone!\n")

    def test_linear(self, x_data):
        # Test our model
        for item in x_data:
            print(item)
        print('->')
        answer = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        for item in answer:
            print(item)
        print('\n')

    def test_sigmoid(self, x_data):
        # Test our model
        for item in x_data:
            print(item)
        print('->')
        answer = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        for item in answer:
            print(item)
        predicted_casted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        print('Casted value: ', self.sess.run(predicted_casted, feed_dict={self.X: x_data}))
        print('\n')

    def test_argmax(self, x_data):
        # Testing & One-hot encoding
        for item in x_data:
            print(item)
        print('-> Hypothesis')
        answer = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        for item in answer:
            print(item)
        print('-> tf.arg_max()')
        print(self.sess.run(tf.arg_max(answer, 1)))
        print('\n')

    def evaluate_linear(self, x_data, y_data):
        # Test our model
        for item in x_data:
            print(item)
        print('->')
        answer = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        for item in answer:
            print(item)
        print('\n')

        accuracy = tf.reduce_mean(tf.cast(tf.equal(answer, self.Y), dtype=tf.float32))
        print('Accuracy: {}%'.format(100*self.sess.run(accuracy, feed_dict={self.Y: y_data})))
        print('\n')

    def evaluate_sigmoid(self, xdata, ydata):
        # Accuracy computation
        # True if hypothesis > 0.5 else False
        predicted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        hit_record = tf.equal(predicted, self.Y)
        accuracy = tf.reduce_mean(tf.cast(hit_record, dtype=tf.float32))

        # Accuracy report
        h, c, a = self.sess.run([self.hypothesis, predicted, accuracy], feed_dict={self.X: xdata, self.Y: ydata})
        print("Predicted(original):\n", h, "\n\nPredicted(casted):\n", c, "\n\nAccuracy: {}".format(a * 100))

    def evaluate_file_one_hot(self, afile, class_num_for_one_hot):
        f2b = File2Buffer()
        f2b.file_load(afile)

        Y_one_hot = tf.one_hot(self.Y, class_num_for_one_hot)  # one hot
        print("one_hot original", Y_one_hot)
        Y_onehot_reshaped = tf.reshape(Y_one_hot, [-1, class_num_for_one_hot])  # 리스트 [[a],[b]] -> [a, b]
        print("one_hot reshaped", self.Y_one_hot)

        prediction = tf.argmax(self.hypothesis, 1) #
        correct_prediction = tf.equal(prediction, tf.argmax(Y_onehot_reshaped, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        acc = self.sess.run(accuracy, feed_dict={self.X: f2b.x_data, self.Y: f2b.y_data})
        print("Acc: {:.2%}".format(acc))

    def xavier(self):
        self.initializer = NNType.XAIVER
        print('Now, we are using Xavier initializer for weights.')

    def dropout(self):
        self.dropout = NNType.DROPOUT
        print('Dropout occurs..')