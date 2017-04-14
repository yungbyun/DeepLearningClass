import matplotlib.pyplot as plt
import tensorflow as tf
from softmax_del import Softmax

from lib import mytool

'''
gildong = MnistClassifier()
gildong.learn(3, 100) # epoch, partial_size
gildong.evaluate() # for all test data
gildong.classify_random_image() # classify a randomly selected image
#gildong.show_errors()

'''
class MnistClassifier (Softmax):
    db = None
    learning_epoch = None #15
    size_of_segment = None #100

    def load_mnist(self):
        return mytool.load_mnist()

    def learn(self, epoch, partial):
        self.learning_epoch = epoch
        self.size_of_segment = partial

        self.db = self.load_mnist()
        super().learn(self.db, self.learning_epoch, self.size_of_segment)

    def get_number_of_segment(self):
        return int(self.db.train.num_examples / self.size_of_segment) #55,000 / 100

    def get_next_segment(self):
        return self.db.train.next_batch(self.size_of_segment)

    def get_image(self, index):
        # Get one and predict
        image = self.db.test.images[index:index+1]
        return image

    def get_label(self, index):
        label = self.db.test.labels[index:index+1]
        return label

    def get_class(self, index):
        label = self.db.test.labels[index:index+1]
        return self.sess.run(tf.arg_max(label, 1))

    def classify(self, an_image):
        category = self.sess.run(tf.argmax(self.hypothesis, 1), feed_dict={self.X: an_image})
        return category

    def classify_random_image(self):
        index = mytool.get_random_int(self.db.test.num_examples)

        image = self.get_image(index)
        label = self.get_class(index)

        category = self.classify(image)
        print('Label', label)
        print('Classified', category)

        self.show_image(image)

    def show_image(self, image):
        plt.imshow(image.reshape(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()

    # 테스트 데이터로 평가
    def evaluate(self):
        # Test model
        is_correct = tf.equal(tf.arg_max(self.hypothesis, 1), tf.arg_max(self.Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        # Test the model using test sets
        result = accuracy.eval(session=self.sess, feed_dict={self.X: self.db.test.images, self.Y: self.db.test.labels})
        #result = self.sess.run(accuracy, feed_dict={self.X: db.test.images, self.Y: db.test.labels})

        print("Recognition rate :", result)
