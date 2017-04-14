from abc import abstractmethod
import tensorflow as tf


class CNNCore:
    X = None
    X_2d  = None
    Y = None

    DO = None

    logit = None #
    hypothesis = None
    cost_function = None
    optimizer = None

    accuracy = None

    @abstractmethod
    def init_network(self):
        pass

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.build_net()

    def build_net(self):
        with tf.variable_scope(self.name):
            self.init_network()

            # 첫번째 문자를 네트워크 입력한 후 출력되는 10개 요소 중 최대값을 갖는 인덱스
            # 해당 정답 라벨 10개 요소 중 최대값을 갖는 인덱스
            # 위 두개가 같으면 1, 아니면 0
            # 문자 10,000개가 입력되면 출력되는 0 혹은 1도 10,000개
            # 네트워크 생성할 때 있어야 하나? evaluate 함수로 옮기니 결과가 조금 달라진다.
            hit_record = tf.equal(tf.argmax(self.hypothesis, 1), tf.argmax(self.Y, 1))
            # 0과 1 10,000개로 구성된 리스트 각각을 실수로 캐스팅한 후 평균을 구함. 이것이 인식률
            self.accuracy = tf.reduce_mean(tf.cast(hit_record, tf.float32))
            # 위 두 줄을 네트워크 생성할 때 작성해야 하나? evaluate 함수로 옮기니 결과가 조금 달라진다. 이상...

    # 입력을 준 후 네트워크의 출력값을 구함.
    def test(self, x_test, keep_prop):
        return self.sess.run(self.logit, feed_dict={self.X: x_test, self.DO: keep_prop})

    # 인식률 구하기
    def evaluate(self, x_test, y_test, keep_prop):
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.DO: keep_prop})

    # 한 세그먼트(100개) 데이터를 이용하여 한번 미분 수행(W, b 조정)
    def learn_with_a_segment(self, x_seg, y_seg, keep_prop):
        return self.sess.run([self.cost_function, self.optimizer], feed_dict={self.X: x_seg, self.Y: y_seg, self.DO: keep_prop})

    ###########

    def set_placeholder(self, input_size, num_of_class, size_x, size_y):
        self.X = tf.placeholder(tf.float32, [None, input_size])
        self.X_2d = tf.reshape(self.X, [-1, size_x, size_y, 1])   # -1은 여러개의 입력을 의미. img 28x28x1 (black/white)
        self.Y = tf.placeholder(tf.float32, [None, num_of_class])

    #컨볼루션 층을 만듦
    def convolution_layer(self, pre_output, filter_x, filter_y, depth, num_of_filter, move_right, move_down):
        # 필터를 32개 만듦 : 3x3x1 짜리 필터
        # L1 ImgIn shape=(?, 28, 28, 1)
        W = tf.Variable(tf.random_normal([filter_x, filter_y, depth, num_of_filter], stddev=0.01)) # 3x3x1 필터 32개
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)

        # 필터 하나를 이용하여 오른쪽으로, 아래로, 가면서 convolution layer plane 하나를 만듦.
        # 모든 필터 32개를 이용하여 plane 32개를 만듦.
        # 결국  convolution layer L1은 32개 plane을 갖는 convolution layer 층 (activation maps라고 불림-32개의 28*28*1 plane)
        conv_layer = tf.nn.conv2d(pre_output, W, strides=[1, move_right, move_down, 1], padding='SAME') #오른쪽으로 1, 아래로 1
        return conv_layer

    # 기존의 층에 relu 만 적용됨
    def relu(self, layer):
        layer = tf.nn.relu(layer) # plane에 있는 각 값에 대하여 relu 적용 -> 모든 plane에 대하여 적용
        return layer

    # 풀링하여 새로운 층을 만듦
    def max_pool(self, layer, kernel_x, kernel_y, move_right, move_down):
        # 2x2 윈도우를 오른쪽으로 2, 아래쪽으로 2씩 움직이면서 윈도우 내에 있는 가장 큰 값을 꺼내어 Pooling layer 만듦.
        layer = tf.nn.max_pool(layer, ksize=[1, kernel_x, kernel_y, 1], strides=[1, move_right, move_down, 1], padding='SAME')
        # 14x14x32 풀링 레이어
        return layer

    def dropout(self, layer):
        return  tf.nn.dropout(layer, keep_prob=self.DO)

    def fully_connected_layer(self, pre_layer, input_size, output_size, w_name):
        #reshaped_input = tf.reshape(pre_layer, [-1, input_size])
        # Final FC 7x7x64 inputs -> 10 outputs
        W = tf.get_variable(w_name, shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.random_normal([output_size]))
        output = tf.matmul(pre_layer, W) + b
        return output


    def set_hypothesis(self, hy):
        self.hypothesis = hy

    def set_cost_function(self):
        # define cost/loss & optimizer
        self.cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.hypothesis, labels = self.Y))

    def set_optimizer(self, l_rate):
        self.optimizer = tf.train.AdamOptimizer(learning_rate = l_rate).minimize(self.cost_function)

