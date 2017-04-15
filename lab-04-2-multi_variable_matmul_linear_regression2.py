# Lab 4 Multi-variable linear regression

from lib.neural_network import NeuralNetwork
from lib.tensor_board_util import TensorBoardUtil
from lib.nntype import NNType


class MyNeuron (NeuralNetwork):
    tbutil = TensorBoardUtil()

    def init_network(self):
        self.set_placeholder(3, 1)

        # 3 to 1
        hypo = self.fully_connected_layer(self.X, 3, 1, 'W', 'b')

        self.set_hypothesis(hypo)
        self.set_cost_function(NNType.SQUARE_MEAN)
        self.set_optimizer(NNType.GRADIENT_DESCENT, l_rate=1e-5)

        self.tbutil.scalar('Error', self.cost_function)
        self.tbutil.merge()

    def create_writer(self):
        self.tbutil.create_writer(self.sess, './tb/lab_04_2_1')

    def do_summary(self, feed_dict):
        self.tbutil.do_summary(self.sess, feed_dict)


x_dat = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_dat = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

gildong = MyNeuron()
gildong.learn(x_dat, y_dat, 2000, 40)
#gildong.print_log()
gildong.test_linear(x_dat)
gildong.show_error()
