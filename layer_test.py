import numpy as np
import tensorflow as tf

X = np.array([[31, 23,  4, 24, 27, 34],
              [18,  3, 25,  0,  6, 35],
              [28, 14, 33, 22, 20,  8]], dtype=np.float32)

X2 = [[31, 23,  4, 24, 27, 34],
     [18,  3, 25,  0,  6, 35],
     [28, 14, 33, 22, 20,  8.]]

W = tf.Variable(tf.random_normal([6, 2]), name = 'w')
b = tf.Variable(tf.random_normal([2]), name = 'b')
output = tf.nn.relu(tf.add(tf.matmul(X, W), b))
print(output)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

print(sess.run(output))


'''
X = [[1., 2], [2, 2]]
Y = [1.]

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
hypothesis = tf.add(tf.matmul(X, W), b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(hypothesis)
print(sess.run(hypothesis))
'''

print('Hello\n')

cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(num_units=3, state_is_tuple=True) #hidden_size
print(cell)
