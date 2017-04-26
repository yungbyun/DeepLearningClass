import tensorflow as tf
import os

'''
CHECK_POINT_DIR = TB_SUMMARY_DIR = './tb/mnist2'
sa = Saver()
sa.restore_network(sess, CHECK_POINT_DIR)

for epoch in range(sa.get_starting_epoch(), training_epochs):
    # other codes
    sa.save(sess, CHECK_POINT_DIR, epoch, global_step)
'''


class NetworkLoader:
    saver = None
    last_location = tf.Variable(0, name='last_epoch')
    start_from = None

    def restore_network(self, sess, dir):
        # Savor and Restore
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(dir)

        # 만일 체크포인트가 있으면 저장되어 있는 것을 로드하라.
        if checkpoint and checkpoint.model_checkpoint_path: # 있으면
            try:
                self.saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            except:
                print("Error on loading existing network weights")
        else:
            print("No existing network weights")

        self.start_from = sess.run(self.last_location)

        # train my model
        print('Start learning from:', self.start_from)

    def get_starting_epoch(self):
        return self.start_from

    def save_network(self, sess, dir, epoch, step):
        print("Saving network...")
        sess.run(self.last_location.assign(epoch + 1))
        # 폴더가 없으면 만들어서, 있으면 있는 폴더에 저장
        if not os.path.exists(dir):
            os.makedirs(dir)

        self.saver.save(sess, dir + "/model", global_step=step)

