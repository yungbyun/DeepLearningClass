import tensorflow as tf
tf.set_random_seed(777)

x_data = [[-2], [-1], [1], [2]]
y_data = [[0], [0], [1], [1]] #correct answer

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#layer(neuron)
W = tf.Variable(tf.random_normal([1, 1]))
b = tf.Variable(tf.random_normal([1]))
hypothesis = tf.sigmoid(tf.matmul(X, W))

cost_function = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost_function)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    sess.run(gradient_descent, feed_dict={X: x_data, Y: y_data})

    if step % 200 == 0:
        cost_val = sess.run(cost_function, feed_dict={X: x_data, Y: y_data})
        print(step, cost_val)
        # w_log.append(sess.run(W)[0])

# True(1) if hypothesis > 0.5 else false(0)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
result = sess.run(predicted, feed_dict={X: [[-7], [3], [5]]})
print("Input: ", [[-7], [3], [5]])
print("Predicted: ", result)

