import tensorflow as tf


class MyNeuralNetwork1:
    x_data = None
    y_data = None

    hypothesis = None
    cost_function = None
    optimizer = None

    sess = None

    def set_data(self, xdata, ydata):
        self.x_data = xdata
        self.y_data = ydata

    def fully_connected_layer(self, num_of_input, num_of_neuron, w_name, b_name):
        W = tf.Variable(tf.random_normal([num_of_input]), name=w_name)
        b = tf.Variable(tf.random_normal([num_of_neuron]), name=b_name)
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

        for step in range(total_loop + 1):
            cost_val, _ = self.sess.run([self.cost_function, self.optimizer])
            if step % check_step == 0:
                print(step, cost_val)


class SimpleNeuron (MyNeuralNetwork1):
    def init_network(self):

        hypo = self.fully_connected_layer(1, 1, 'W', 'b')

        self.set_hypothesis(hypo)
        self.set_cost_function()
        self.set_optimizer(0.1)


gildong = SimpleNeuron()

x_data =[1, 2, 3]
y_data = [1, 2, 3]

gildong.set_data(x_data, y_data)
gildong.learn(2000, 20)
gildong.test()


