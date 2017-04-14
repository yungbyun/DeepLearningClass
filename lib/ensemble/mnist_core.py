
class MnistCore:
    db = None

    def load_mnist(self):
        from lib import mytool
        self.db = mytool.load_mnist()

    def get_number_of_segment(self, seg_size):
        return int(self.db.train.num_examples / seg_size)  # 55,000 / 100

    def get_next_segment(self, seg_size):
        return self.db.train.next_batch(seg_size)

    def get_test_x_data(self):
        return self.db.test.images

    def get_test_y_data(self):
        return self.db.test.labels
