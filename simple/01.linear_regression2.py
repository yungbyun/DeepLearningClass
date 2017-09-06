import tensorflow as tf
tf.set_random_seed(777)

X = [1, 2, 3]
Y = [1, 2, 3] #correct answer

#layer(neuron)
W = tf.Variable(tf.random_normal([1]))
hypothesis = W * X

cost_function = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.02)
gradient_descent = optimizer.minimize(cost_function)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_log = []
w_log = []

# Learning: find w
for step in range(20):
    sess.run(gradient_descent)
    w_val = sess.run(W)
    cost_val = sess.run(cost_function)
    print(step, 'W=', w_val, 'cost=', cost_val)

    cost_log.append(cost_val)
    w_log.append(w_val)

import matplotlib.pyplot as plot
plot.plot(w_log, 'o-')
plot.show()

