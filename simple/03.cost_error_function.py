import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for reproducibility

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)
hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

w_log = []
cost_log = []

for i in range(-30, 50):
    aa = i * 0.1
    cost_val, w_val = sess.run([cost, W], feed_dict={W: aa})
    print(cost_val, w_val)

    w_log.append(w_val)
    cost_log.append(cost_val)

# Show the cost function
plt.plot(w_log, cost_log)
plt.show()
