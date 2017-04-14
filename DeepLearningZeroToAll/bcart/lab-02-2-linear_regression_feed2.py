# Lab 2 Linear Regression
import tensorflow as tf


class MyNeuralNetwork2:
    # place holders
    X = None
    Y = None

    hypothesis = None
    cost_function = None
    optimizer = None

    sess = None

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

        for step in range(total_loop + 1):
            cost_val, _ = self.sess.run([self.cost_function, self.optimizer],
                feed_dict={self.X: xdata, self.Y: ydata})
            if step % check_step == 0:
                print(step, cost_val)


class XXX (MyNeuralNetwork2):
    def init_network(self):
        self.set_placeholder()

        hypo = self.fully_connected_layer(self.X, 1, 1, 'W', 'b')

        self.set_hypothesis(hypo)
        self.set_cost_function()
        self.set_optimizer(0.01)



gildong = XXX()
gildong.learn([1, 2, 3], [1, 2, 3], 2000, 20)
gildong.learn([1, 2, 3, 4, 5], [2.1, 3.1, 4.1, 5.1, 6.1], 2000, 20)
gildong.test([5])
gildong.test([2.5])
gildong.test([1.5, 3.5])

'''
1980 2.82812e-07 [ 1.00034416] [ 1.09875762]
2000 2.46997e-07 [ 1.00032163] [ 1.09883893]
[ 6.1004467]
[ 3.59964275]
[ 2.59932137  4.59996462]
'''