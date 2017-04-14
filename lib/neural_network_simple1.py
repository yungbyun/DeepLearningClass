import tensorflow as tf
from lib.myplot import MyPlot


class NeuralNetworkSimple1:
    x_data = None
    y_data = None

    hypothesis = None
    cost_function = None
    optimizer = None

    sess = None

    costs = []

    def set_data(self, xdata, ydata):
        self.x_data = xdata
        self.y_data = ydata

    def fully_connected_layer(self, input_num, output_num, w_name, b_name):
        W = tf.Variable(tf.random_normal([input_num]), name=w_name)
        b = tf.Variable(tf.random_normal([output_num]), name=b_name)
        output = self.x_data * W + b
        return output

    def set_hypothesis(self, h):
        self.hypothesis = h

    def set_cost_function(self):
        self.cost_function = tf.reduce_mean(tf.square(self.hypothesis - self.y_data))

    def set_optimizer(self, l_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        self.optimizer = optimizer.minimize(self.cost_function)

    def test(self):
        print(self.sess.run(self.hypothesis))

    def learn(self, total_loop, check_step):
        tf.set_random_seed(777)  # for reproducibility

        if self.sess == None:
            self.init_network()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

        print("\nStart learning:")

        for step in range(total_loop + 1):
            cost_val, _ = self.sess.run([self.cost_function, self.optimizer])
            if step % check_step == 0:
                self.costs.append(cost_val)

                from lib import mytool
                mytool.print_dot()

        print("\nDone!\n")

    def show_error(self):
        mp = MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.costs)