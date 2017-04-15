# Lab 12 RNN
from lib.rnn_core import RNNCore
from lib.tensor_board_util import TensorBoardUtil


class XXX (RNNCore):
    tbutil = TensorBoardUtil()

    def init_network(self):
        self.set_placeholder(seq_len=6, hidden_size=5)

        logits = self.rnn_lstm_cell(hidden_size=5, batch_size=1)

        self.set_hypothesis(logits)
        self.set_cost_function(batch_size=1, seq_len=6)
        self.set_optimizer(0.1)

        self.tbutil.scalar('Cost', self.cost_function)
        self.tbutil.merge()

    def create_writer(self):
        self.tbutil.create_writer(self.sess, './tb/rnn01')

    def do_summary(self, feed_dict):
        self.tbutil.do_summary(self.sess, feed_dict)


x_data = [[0, 1, 0, 2, 3, 3]]  # hihell

x_one_hot = [[[1, 0, 0, 0, 0],   # 0 h
              [0, 1, 0, 0, 0],   # 1 i
              [1, 0, 0, 0, 0],   # 0 h
              [0, 0, 1, 0, 0],   # 2 e
              [0, 0, 0, 1, 0],   # 3 l
              [0, 0, 0, 1, 0]]]  # 3 l

y_data = [[1, 0, 2, 3, 3, 4]]  # ihello

gildong = XXX()
gildong.learn(x_one_hot, y_data)
gildong.test(x_one_hot) # hihell -> ihello
#gildong.print_log()
