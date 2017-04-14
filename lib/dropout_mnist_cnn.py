import tensorflow as tf

from lib.mnist_cnn import MnistCNN


class DropoutMnistCNN(MnistCNN):
    DO = None

    def my_log(self, i, xdata, ydata):
        err = self.sess.run(self.cost_function, feed_dict={self.X: xdata, self.Y: ydata, self.DO: 0.7})
        msg = "Step:{}, Error:{:.6f}".format(i, err)
        self.logs.append(msg)

    def set_placeholder(self, input_size, num_of_class, size_x, size_y):
        super().set_placeholder(input_size, num_of_class, size_x, size_y)

        # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
        self.DO = tf.placeholder(tf.float32)

    def dropout(self, prev):
        dropouted = tf.nn.dropout(prev, keep_prob=self.DO)
        return dropouted

    # MNIST와 같은 데이터를 이용한 학습
    def learn_with_segment(self, db, learning_epoch, partial_size):
        tf.set_random_seed(777)  # for reproducibility

        self.init_network()  # 가상함수

        self.sess = tf.Session()
        # Initialize TensorFlow variables
        self.sess.run(tf.global_variables_initializer())

        print("\nStart learning:")
        # Training cycle
        for epoch in range(learning_epoch):
            err_4_all_data = 0
            number_of_segment = self.get_number_of_segment()  # 가상함수

            # 처음 데이터를 100개를 읽어 최적화함.
            # 그 다음 100개 데이터에 대하여 수행.
            # 이를 모두 550번 수행하면 전체 데이터 55,000개에 대해 1번 수행하게 됨.
            # 아래 for 문장이 한번 모두 실행되면 전체 데이터에 대해 1번 실행(학습)함.
            for i in range(number_of_segment):
                x_data, y_data = self.get_next_segment()  # 가상함수

                # 아래 에러는 일부분(100개)에 대한 것이므로 전체 에러를 구하려면 550으로 나누어주어야 함. 아래에서 수행
                err_4_partial, _ = self.sess.run([self.cost_function, self.optimizer],
                                                 feed_dict={self.X: x_data, self.Y: y_data, self.DO: 0.7})
                err_4_all_data += err_4_partial

            from lib import mytool
            mytool.print_dot()
            avg_err = err_4_all_data / number_of_segment  #
            self.costs.append(avg_err)

            self.my_log(epoch, x_data, y_data)  # 가상함수

        print("\nDone!\n")

    def classify(self, mnist_image):
        category = self.sess.run(tf.argmax(self.hypothesis, 1), feed_dict={self.X: mnist_image, self.DO: 1})
        return category

    # 테스트 데이터로 평가
    def evaluate(self):
        # Test model
        is_correct = tf.equal(tf.arg_max(self.hypothesis, 1), tf.arg_max(self.Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        # Test the model using test sets
        result = accuracy.eval(session=self.sess,
                               feed_dict={self.X: self.db.test.images, self.Y: self.db.test.labels, self.DO: 1})
        # result = self.sess.run(accuracy, feed_dict={self.X: db.test.images, self.Y: db.test.labels})

        print("Recognition rate :", result)