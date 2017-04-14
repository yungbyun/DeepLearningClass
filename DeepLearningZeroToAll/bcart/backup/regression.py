import tensorflow as tf

from lib import myplot


class Regression:
    costs = []
    weights = []
    biases = []
    logs = []

    def show_error(self):
        mp = myplot.MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.costs)

    def show_weight(self):
        mp = myplot.MyPlot()
        mp.set_labels('Step', 'Weight')
        mp.show_list(self.weights)

    def show_bias(self):
        mp = myplot.MyPlot()
        mp.set_labels('Step', 'Bias')
        mp.show_list(self.biases)

    def print_log(self):
        for item in self.logs:
            print(item)

    # tf Graph Input
    X = tf.placeholder("float")
    Y = tf.placeholder("float")
