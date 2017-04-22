# RNN에서 인식하고자 하는 문장을 입력받아 중복문자 제거, 문자를 인덱스로, 인덱스를 문자로 변환해주는 모듈
class SentenceTool:
    unique_char_list = []
    unique_char_index_set = {}

    def set_sentence(self, s):
        self.unique_char_list = list(set(s))
        self.unique_char_index_set = {w: i for i, w in enumerate(self.unique_char_list)}

    def unique_char_num(self):
        return len(self.unique_char_list)

    # 문자열을 줄테니 인덱스 리스트를 다오.
    def string_to_index(self, str1):
        x_index_list = [self.unique_char_index_set[c] for c in str1]  # x str to index
        return x_index_list

    def index_to_string(self, idx):
        s = [self.unique_char_list[t] for t in idx]
        return s
