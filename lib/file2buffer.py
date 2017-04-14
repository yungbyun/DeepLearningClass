import numpy as np

'''
가장 마지막에 있는 하나가 출력(y)
출력을 제외한 앞의 모든 데이터가 입력(x)인 데이터 파일을 읽어온다.
'''


class File2Buffer:
    x_data = None
    y_data = None

    x_dimension = None
    y_dimension = None

    def file_load(self, filename):
        xy = np.loadtxt(filename, delimiter=',', dtype=np.float32)

        #print('shape', xy.shape, 'size', xy.size, 'itemsize', xy.itemsize)
        self.x_data = xy[:, 0:-1] #모든 행을 선택, 0번째부터 마지막을 빼고 읽어라.
        self.y_data = xy[:, [-1]] # 맨 마지막은

        self.x_data_len = len(self.x_data)
        self.x_data_col = len(self.x_data[0])
        self.y_data_len = len(self.y_data)
        self.y_data_col = len(self.y_data[0])

    def print_info(self):
        print(self.x_data.shape, self.x_data_len, self.x_data_col)
        print(self.y_data.shape, self.y_data_len, self.y_data_col)

    def get_x_dimension(self):
        return self.x_dimension

    def get_y_dimension(self):
        return self.y_dimension

