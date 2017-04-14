import tensorflow as tf

from lib.neural_network import NeuralNetwork


class SoftMaxClassifier (NeuralNetwork):
    Y_one_hot = None

    def set_placeholder(self, x_dim, y_dim):
        self.X = tf.placeholder(tf.float32, shape=[None, x_dim])
        self.Y = tf.placeholder(tf.int32, shape=[None, y_dim])

    def set_one_hot(self, num_of_class):
        self.Y_one_hot = tf.one_hot(self.Y, num_of_class)  # Y를 줄테니 원 핫으로 바꿔주라
        print("one_hot", self.Y_one_hot)
        self.Y_one_hot = tf.reshape(self.Y_one_hot, [-1, num_of_class]) #리스트 [[a],[b]] -> [a, b]
        print("reshape", self.Y_one_hot)
    '''
    def set_cost_function(self, type):
        super().set_cost_function(type)
        if type == MyType.SOFTMAX_LOGITS:
            # Cross entropy cost/loss
            logits = tf.matmul(self.X, self.W) + self.b
            # !!!! 이런일이.. 아래 코드를 Base 클래스로 옮겨도 문제없다. NeuralNetwork에서 SoftMaxClassification의 멤버 Y_one_hot을 액세스한다!!!
            cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.Y_one_hot)
            self.cost_function = tf.reduce_mean(cost_i)
    '''
