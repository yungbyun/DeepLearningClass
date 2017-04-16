import numpy as np


class SentenceToIndex:
    unique_char_list = []
    unique_char_and_index = []

    def set_sentence(self, sentence):
        unique_char_collec = set(sentence)  # set class는 중복된 문자(space 3개, y, o, u)를 제거한 후 무작위로 collection 생성
        # tmp = {'n', 't', 'y', 'w', ' ', 'f', 'u', 'a', 'i', 'o'}

        self.unique_char_list = list(unique_char_collec)  # index -> char
        # uique_char_list = ['n', 't', 'y', 'w', ' ', 'f', 'u', 'a', 'i', 'o']

        aa = enumerate(self.unique_char_list)

        # {'y': 0, 'a': 1, 'f': 2, 'o': 3, 'i': 4, 'w': 5, 't': 6, 'u': 7, 'n': 8, ' ': 9}
        self.unique_char_and_index = {c: i for i, c in aa}
        #print(unique_char_and_index)

    def index_list_to_sentence(self, index_list):
        str = [self.unique_char_list[c] for c in np.squeeze(index_list)]
        str = ''.join(str) #list to string
        return str

    def sentence_to_index_list(self, sentence):
        # 샘플 문장에 있는 문자 순서대로 인덱스를 구함
        # ' if you want you' 문장 전체에 있는 문자 인덱스 리스트
        char_index_list = [self.unique_char_and_index[c] for c in sentence]  # char to index
        # [7, 1, 3, 7, 6, 5, 9, 7, 8, 0, 4, 2, 7, 6, 5, 9]

        return char_index_list

