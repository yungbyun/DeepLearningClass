import numpy as np


def squeeze():
    idx2char = ['h', 'i', 'e', 'l', 'o']

    result_str = [idx2char[c] for c in np.squeeze([[1,0,2,3,3,3]])]
    print('[[1,0,2,3,3,3]]', '-->', ''.join(result_str))

