import sys

#학습하는 동안 진행되는 것을 표시하기 위하여 점을 한줄로 찍도록 하는 코드. print('.')는 새로운 줄에 찍어버림.
def write(str):
    sys.stdout.write('.')
    sys.stdout.flush()

def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    # Check out https://www.tensorflow.org/get_started/mnist/beginners for more information about the mnist dataset
    return input_data.read_data_sets("MNIST_data/", one_hot=True)

