import tensorflow as tf
tf.set_random_seed(777)

X = [1, 2, 3]
Y = [1, 2, 3]

#layer(neuron)
W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))
hypothesis = W * X + b

cost_function = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
gradient_descent = optimizer.minimize(cost_function)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Learning: find w and b
for step in range(1000):
    sess.run(gradient_descent)

    if step % 20 == 0:
        w_val = sess.run(W)
        b_val = sess.run(b)
        print(step, 'w:', w_val, 'b:', b_val)

