import numpy as np
from abc import abstractmethod


class CommonDB:
    native_data = []

    # 시계열 데이터를 학습용(70), 테스트용(30)으로 분리
    trainX = []
    trainY = []

    testX = []
    testY = []

    @abstractmethod
    def preprocesisng(self):
        pass

    def min_max_scalr(self):
        numerator = self.native_data - np.min(self.native_data, 0)
        denominator = np.max(self.native_data, 0) - np.min(self.native_data, 0)
        # noise term prevents the zero division
        self.native_data = numerator / (denominator + 1e-7)

    # in: native_data, out: native_data
    def reverse(self):
        # reverse order (chronically ordered) 맨 뒷 데이터를 처음으로...
        self.native_data = self.native_data[::-1]

    # in: native_data, out: native_data
    # 5개의 열 별로 0~1 정규화 수행
    def normalize(self): # np.shape(data) = (732, 5)
        print(np.shape(self.native_data))

        num_row = np.shape(self.native_data)[0] # 732
        num_col = np.shape(self.native_data)[1] # 5
        array = np.zeros((num_row, num_col)) #
        for i in range(num_col): #5
            input = self.native_data[:, i] # 0번째, 1번째, ... 4번째 열 통째로..
            array[:, i] = (input - np.min(input)) / (np.max(input) - np.min(input))
        self.native_data = array

    # 로드, 순서바꿈, 정규화, 시계열 데이터 생성, 학습과 테스트용 데이터로 분리
    def load(self, file, seq_length):
        # Open,High,Low,Close,Volume  -> (732, 5)
        self.native_data = np.loadtxt(file, delimiter=',')

        self.preprocesisng()  # a hole for you!

        originalX = self.native_data # 전체
        originalY = self.native_data[:, [-1]]  # Close as label, # 전체에서 마지막 것(폐장 주식 가격)만 선택.
        # print(len(self.x)) # 전체 라인 수 = 732 (0~731)

        #cut_and_append, 시계열 데이터
        dataX = []
        dataY = []
        # append 횟수는? 전체 줄 수(732) - sequence length(7) = 725
        for i in range(0, len(originalY) - seq_length):
            _x = originalX[i:i + seq_length]
            _y = originalY[i + seq_length]  # Next close price as target
            #print(_x, "->", _y)
            dataX.append(_x)
            dataY.append(_y)

        # split to train and testing
        train_size = int(len(dataY) * 0.7)
        test_size = len(dataY) - train_size
        self.trainX, self.testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
        self.trainY, self.testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

        print('trainX: {} to {}'.format(0, train_size))
        print('trainY: {} to {}'.format(0, train_size))

