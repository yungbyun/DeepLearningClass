import numpy as np
import tensorflow as tf

# 문자열을 입력받아 RNN에서 학습하는데 필요한 정보와 데이터를 만들어 줌.

class StringDB3:
    unique_char_list = []
    unique_char_and_index = []

    x_index_list = []
    y_index_list = []

    unique_char_num = 0
    sequence_num = 0

    x_one_hot = None
    y_one_hot = None

    def load(self, string):
        self.make_unique_lists(string)

        self.unique_char_num = len(self.unique_char_list)
        self.sequence_num = len(string) - 1

        str_len = len(string)
        xd = string[0:str_len-1]
        yd = string[1:str_len]

        self.x_index_list = self.sentence_to_index_list(xd)
        # [1, 4, 1, 0, 3, 3]
        self.y_index_list = self.sentence_to_index_list(yd)
        # [4, 1, 0, 3, 3, 2]

        self.x_one_hot = tf.one_hot(self.x_index_list, self.unique_char_num)
        self.y_one_hot = tf.one_hot(self.y_index_list, self.unique_char_num)

    # 문장을 주면 문자 중복 제거 후 문자와 인덱스 쌍을 만듦.
    def make_unique_lists(self, sentence):
        unique_char_collec = set(sentence)  # set class는 중복된 문자(space 3개, y, o, u)를 제거한 후 무작위로 collection 생성
        # tmp = {'n', 't', 'y', 'w', ' ', 'f', 'u', 'a', 'i', 'o'}

        self.unique_char_list = list(unique_char_collec)  # index -> char
        # uique_char_list = ['n', 't', 'y', 'w', ' ', 'f', 'u', 'a', 'i', 'o']

        aa = enumerate(self.unique_char_list)

        # {'y': 0, 'a': 1, 'f': 2, 'o': 3, 'i': 4, 'w': 5, 't': 6, 'u': 7, 'n': 8, ' ': 9}
        self.unique_char_and_index = {c: i for i, c in aa}
        #print(unique_char_and_index)

    # 인덱스 리스트를 주면 문자 리스트를 생성해줌.
    def index_list_to_sentence(self, index_list):
        str = [self.unique_char_list[c] for c in np.squeeze(index_list)]
        str = ''.join(str) #list to string
        return str

    # 문장을 주면 인덱스 리스트를 생성해 줌.
    def sentence_to_index_list(self, sentence):
        # 샘플 문장에 있는 문자 순서대로 인덱스를 구함
        # ' if you want you' 문장 전체에 있는 문자 인덱스 리스트
        char_index_list = [self.unique_char_and_index[c] for c in sentence]  # char to index
        # [7, 1, 3, 7, 6, 5, 9, 7, 8, 0, 4, 2, 7, 6, 5, 9]

        return char_index_list

    def get_batch_size_and_sequence_length(self, sentence):
        batch_size = 1  # one sample data, one batch
        sequence_length = len(sentence) - 1  # 16 - 1 = 15, number of lstm rollings (unit #)
        return batch_size, sequence_length

    def get_unique_char_num(self, my_sentence):
        self.make_unique_lists(my_sentence)

        number_of_unique_char = len(self.unique_char_and_index)  # 10, RNN output size
        #가령, 고유한 문자 수가 10개이면 10비트(그중 하나만 1인, one_hot)로 입력과 출력을 표현함.
        #따라서 출력수, 클래스 수, 입력 수 모두 10이됨.

        return number_of_unique_char

