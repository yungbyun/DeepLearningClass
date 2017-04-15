import tensorflow as tf

'''
How to use
1. NeuralNetwork를 상속받는 클래스에서 멤버변수 정의
2. init_network에서 scalar 등을 호출한 후 마지막에 merge 호출
3. create_writer 오버로딩 후 여기의 create_writer 호출 (폴더명의 예: './tb/mnist')
4. do_summary 오버로딩 후 여기의 do_summary 호출 


'''
class TensorBoardUtil:
    summary = None
    global_step = 0;
    writer = None

    def scalar(self, label, target):
        tf.summary.scalar(label, target)
        print('scalar')

    def merge(self):
        self.summary = tf.summary.merge_all()

    def create_writer(self, sess, dir):
        # Create summary writer
        self.writer = tf.summary.FileWriter(dir)
        self.writer.add_graph(sess.graph)
        self.global_step = 0
        print('$ tensorboard --logdir [dir] ex)./tb')

    def do_summary(self, sess, feed):
        s = sess.run(self.summary, feed_dict=feed)
        self.writer.add_summary(s, global_step=self.global_step)
        self.global_step += 1
