import tensorflow as tf
from lib.myplot import MyPlot


class NeuralNetworkSimple2:
    # place holders
    X = None
    Y = None

    hypothesis = None
    cost_function = None
    optimizer = None

    sess = None

    costs = []

    def set_placeholder(self):
        self.X = tf.placeholder(tf.float32, shape=[None])
        self.Y = tf.placeholder(tf.float32, shape=[None])

    def fully_connected_layer(self, previous_output, num_of_input, num_of_neuron, w_name, b_name):
        W = tf.Variable(tf.random_normal([num_of_input]), name=w_name)
        b = tf.Variable(tf.random_normal([num_of_neuron]), name=b_name)
        output = previous_output * W + b
        return output

    def set_hypothesis(self, h):
        self.hypothesis = h

    def set_cost_function(self):
        self.cost_function = tf.reduce_mean(tf.square(self.hypothesis - self.Y))

    def set_optimizer(self, l_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.optimizer = optimizer.minimize(self.cost_function)

    def test(self, xdata):
        print(self.sess.run(self.hypothesis, feed_dict={self.X: xdata}))

    def learn(self, xdata, ydata, total_loop, check_step):
        tf.set_random_seed(777)  # for reproducibility

        if self.sess == None:
            self.init_network()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

        print("\nStart learning:")

        for step in range(total_loop + 1):
            cost_val, _ = self.sess.run([self.cost_function, self.optimizer],
                feed_dict={self.X: xdata, self.Y: ydata})
            if step % check_step == 0:
                self.costs.append(cost_val)

                from lib import mytool
                mytool.print_dot()

        print("\nDone!\n")

    def show_error(self):
        mp = MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.costs)