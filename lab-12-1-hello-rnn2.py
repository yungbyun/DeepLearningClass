# Lab 12 RNN
from lib.rnn_core import RNNCore
from lib.tensor_board_util import TensorBoardUtil


class XXX (RNNCore):
    tbutil = TensorBoardUtil()

    def init_network(self):
        self.set_placeholder(seq_len=6, hidden_size=5)

        logits = self.rnn_lstm_cell2(self.X, num_classes=5, hidden_size=5, batch_size=1)

        self.set_hypothesis(logits)
        self.set_cost_function(batch_size=1, seq_len=6)
        self.set_optimizer(0.1)

        #self.tbutil.scalar('Cost', self.cost_function)
        #self.tbutil.merge()

    def create_writer(self):
        #self.tbutil.create_writer(self.sess, './tb/rnn01')
        pass

    def do_summary(self, feed_dict):
        #self.tbutil.do_summary(self.sess, feed_dict)
        pass


gildong = XXX()

gildong.set_parameters('hihello')
gildong.show_parameters()

xd = 'hihell'
yd = 'ihello'
gildong.learn(xd, yd, 500, 10)
gildong.predict('hihell') # hihell -> ihello
gildong.predict('hehell') # (1, 6)
gildong.show_error()

