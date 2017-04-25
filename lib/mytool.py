import sys
import random
import numpy as np


# 열 별로 0~1 정규화 수행
def min_max_scalar(self, data):  # np.shape(data) = (732, 5)
    print(np.shape(data))

    num_row = np.shape(data)[0]  # 732
    num_col = np.shape(data)[1]  # 5
    array = np.zeros((num_row, num_col))  #
    for i in range(num_col):  # 5
        input = data[:, i]  # 0번째, 1번째, ... 4번째 컬럼 통째
        array[:, i] = (input - np.min(input)) / (np.max(input) - np.min(input))
    return array

#학습하는 동안 진행되는 것을 점으로 표시.
def print_dot():
    sys.stdout.write('.')
    sys.stdout.flush()

def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    # Check out https://www.tensorflow.org/get_started/mnist/beginners for more information about the mnist dataset
    return input_data.read_data_sets("MNIST_data/", one_hot=True)

def get_random_int(max):
    return random.randint(0, max - 1)

def printf():
    msg = "{} {:.6f} {} {}".format(1, 0.693147, 9.64292212e-07, 9.65349955e-07)
    print(msg)

def get_numpy_data():
    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
    return x_data, y_data

def tensor_version():
    import tensorflow as tf
    print(tf.__version__) #0.12.1

def tensorboard_msg():
    print('http://10.0.1.4:6006')

    '''
    tensorboard --logdir tb/
    Starting TensorBoard b'41' on port 6006
    (You can navigate to http://10.0.1.4:6006)

    '''

'''
tensor_version()

x_one_hot = [[1, 0, 0, 0, 0],   # 0 h
              [0, 1, 0, 0, 0],   # 1 i
              [1, 0, 0, 0, 0],   # 0 h
              [0, 0, 1, 0, 0],   # 2 e
              [0, 0, 0, 1, 0],   # 3 l
              [0, 0, 0, 1, 0]]  # 3 l
y_data = [1, 0, 2, 3, 3, 4]  # ihello

for x, y in zip(x_one_hot, y_data):
    print(x, ':', y,'\n')


# x_col = len(x_data[0])
# y_col = len(y_data[0])
# print(x_col, y_col) # 3, 1


h = [1, 0, 0, 0]
x_data = np.array([[h]], dtype=np.float32) # x_data = [[[1,0,0,0]]] #rank 3,

pp = pprint.PrettyPrinter(indent=4)


x_data = np.arange(45, dtype=np.float32).reshape(batch_size, sequence_length, input_dim)
print(x_data)

[[[  0.   1.   2.]
  [  3.   4.   5.]
  [  6.   7.   8.]
  [  9.  10.  11.]
  [ 12.  13.  14.]]

 [[ 15.  16.  17.]
  [ 18.  19.  20.]
  [ 21.  22.  23.]
  [ 24.  25.  26.]
  [ 27.  28.  29.]]

 [[ 30.  31.  32.]
  [ 33.  34.  35.]
  [ 36.  37.  38.]
  [ 39.  40.  41.]
  [ 42.  43.  44.]]]

'''

