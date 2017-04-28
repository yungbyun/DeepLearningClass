# Lab 12 RNN
from lib.rnn_core import RNNCore
from lib.tensor_board_util import TensorBoardUtil


class XXX (RNNCore):
    tbutil = TensorBoardUtil()

    def init_network(self):
        self.set_placeholder(seq_len=self.length_of_sequence, hidden_size=self.hidden_size)

        logits = self.rnn_lstm_cell2(self.X, num_classes=self.input_size, hidden_size=self.hidden_size, batch_size=1)

        self.set_hypothesis(logits)
        self.set_cost_function(batch_size=1, seq_len=self.length_of_sequence)
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

gildong.set_parameters(' hello,world!') #
gildong.show_parameters()

xd = ' hello,world'
yd = 'hello,world!'
gildong.learn(xd, yd, 1000, 10)
gildong.predict(' hello,world') # hihell -> ihello
#gildong.predict(' heel') # (1, 6)
gildong.show_error()

