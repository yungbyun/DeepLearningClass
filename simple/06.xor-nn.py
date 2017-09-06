import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([[0], [1], [1], [0]])

X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

#layer1
W1 = tf.Variable(tf.random_normal([2, 2]))
b1 = tf.Variable(tf.random_normal([2]))
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

#layer2
W2 = tf.Variable(tf.random_normal([2, 1]))
b2 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    sess.run(gradient_descent, feed_dict={X: x_data, Y: y_data})

    if step % 100 == 0:
        print('step:', step, 'cost:', sess.run(cost, feed_dict={X: x_data, Y: y_data}))


# Test: true(1) if hypothesis > 0.5 else false(0)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
print('Input:', x_data)
result = sess.run(predicted, feed_dict={X: x_data})
print("Predicted: ", result)

