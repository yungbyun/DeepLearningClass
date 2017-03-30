import tensorflow as tf
from myplot import MyPlot
from abc import abstractmethod
from mytype import MyType
import mytool

class NeuralNetwork:
    # place holders
    X = None
    Y = None

    hypothesis = None
    cost_function = None

    sess = None

    costs = []
    weights = []
    biases = []

    def set_placeholder(self, num_of_input, num_of_output):
        self.X = tf.placeholder(tf.float32, [None, num_of_input])
        self.Y = tf.placeholder(tf.float32, [None, num_of_output])

    def create_layer(self, previous_output, num_of_input, num_of_neuron, w_name, b_name, hypothesis_type):
        W = tf.Variable(tf.random_normal([num_of_input, num_of_neuron]), name=w_name)
        b = tf.Variable(tf.random_normal([num_of_neuron]), name=b_name)

        if previous_output is None:
            if hypothesis_type == MyType.LINEAR:
                pass
            elif hypothesis_type == MyType.LOGISTIC:
                output = tf.sigmoid(tf.matmul(self.X, W) + b)
        else:
            if hypothesis_type == MyType.LINEAR:
                pass
            elif hypothesis_type == MyType.LOGISTIC:
                output = tf.sigmoid(tf.matmul(previous_output, W) + b)

        return W, b, output

    def set_hypothesis(self, h):
        self.hypothesis = h

    def set_cost_function(self, type):
        if type == MyType.LINEAR: #linear
            self.cost_function = tf.reduce_mean(tf.square(self.hypothesis - self.Y))
        elif type == MyType.LOGISTIC: #logistic
            self.cost_function = -tf.reduce_mean(self.Y * tf.log(self.hypothesis) + (1 - self.Y) * tf.log(1 - self.hypothesis))
        elif type == MyType.SOFTMAX: #softmax
            # Cross entropy cost/loss
            self.cost_function = tf.reduce_mean(-tf.reduce_sum(self.Y * tf.log(self.hypothesis), axis=1))

    def set_optimizer(self, l_rate):
        self.optimizer = tf.train.GradientDescentOptimizer(l_rate).minimize(self.cost_function)

    def evaluate(self, xdata, ydata):
        # Accuracy computation
        # True if hypothesis>0.5 else False
        predicted_casted = tf.cast(self.hypothesis > 0.5, dtype=tf.float32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_casted, self.Y), dtype=tf.float32))

        # Accuracy report
        h, c, a = self.sess.run([self.hypothesis, predicted_casted, accuracy], feed_dict={self.X: xdata, self.Y: ydata})
        print("\nPredicted(original):\n", h, "\nPredicted(casted):\n", c, "\nAccuracy: ", a)

    @abstractmethod
    def init_network(self):
        pass

    @abstractmethod
    def my_log(self, i, xdata, ydata):
        pass

    def learn(self, x_data, y_data, total_loop, check_step):
        tf.set_random_seed(777)

        self.init_network()  # override

        # Launch graph
        self.sess = tf.Session()
        # Initialize TensorFlow variables
        self.sess.run(tf.global_variables_initializer())

        print('\nStart learning.')
        for i in range(total_loop + 1):  # 10,001
            self.sess.run(self.optimizer, feed_dict={self.X: x_data, self.Y: y_data})

            if i % check_step == 0:
                mytool.print_dot()
                err = self.sess.run(self.cost_function, feed_dict={self.X: x_data, self.Y: y_data})
                self.costs.append(err)
                # self.sess.run([W1, W2])
                self.my_log(i, x_data, y_data)  # override

        print('\nEnded!\n')

    def show_error(self):
        mp = MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.costs)

    def test(self, x_data):
        # Test our model
        for item in x_data:
            print(item)
        print('->')
        answer = self.sess.run(self.hypothesis, feed_dict={self.X: x_data})
        for item in answer:
            print(item)
        print('\n')