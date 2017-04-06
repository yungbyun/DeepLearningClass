import tensorflow as tf
from abc import abstractmethod
from myplot import MyPlot
from nntype import NNType

'''
메인함수인 learn을 호출하기 전에 다음 가상함수 오버라이딩 해야 함
1. init_network
2. get_number_of_partial
3. epoch_process -> optional
'''
class Softmax:
    X = None
    Y = None

    W = None
    b = None

    hypothesis = None
    cost_function = None
    optimizer = None

    sess = None

    errors = []
    logs =  []

    def set_placeholder(self, num_of_input, num_of_neuron):
        self.X = tf.placeholder(tf.float32, [None, num_of_input])
        self.Y = tf.placeholder(tf.float32, [None, num_of_neuron])

    def set_weight_bias(self, num_of_input, num_of_neuron):
        self.W = tf.Variable(tf.random_normal([num_of_input, num_of_neuron]))
        self.b = tf.Variable(tf.random_normal([num_of_neuron]))

    def set_hypothesis(self, type):
        if type == NNType.SOFTMAX:
            # Hypothesis (using softmax)
            self.hypothesis = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)
        elif type == NNType.LOGITS:
            self.hypothesis = tf.matmul(self.X, self.W) + self.b

    def set_cost_function(self, type):
        if type == NNType.SOFTMAX:
            self.cost_function = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.hypothesis), axis=1))
        elif type == NNType.SOFTMAX_LOGITS:
            # hypothesis = logits -> tf.matmul(self.X, self.W) + self.b
            # self.hypothesis = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)
            # define cost/loss & optimizer
            self.cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.hypothesis, labels=self.Y))

    def set_optimizer(self, type, l_rate):
        if type == NNType.GRADIENT_DESCENT:
            self.optimizer = tf.train.GradientDescentOptimizer(l_rate).minimize(self.cost_function)
        elif type == NNType.ADAM:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self.cost_function)

    @abstractmethod
    def init_network(self):
        pass

    def epoch_process(self, avg_err, x_data, y_data):
        pass

    @abstractmethod
    def get_number_of_segment(self):
        pass

    @abstractmethod
    def get_next_segment(self):
        pass


    '''

    '''
    def learn(self, db, learning_epoch, partial_size):
        tf.set_random_seed(777)  # for reproducibility

        self.init_network()  # 가상함수

        self.sess = tf.Session()
        # Initialize TensorFlow variables
        self.sess.run(tf.global_variables_initializer())

        print("\nStart learning.")
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
            self.errors.append(avg_err)

            self.epoch_process(avg_err, x_data, y_data)  # 가상함수

        print("\nEnded!\n")

    def show_errors(self):
        mp = MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.errors)

    def print_log(self):
        for item in self.logs:
            print(item)