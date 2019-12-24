import pickle
import os
# from dataset import char
from config import hparams as hp

_pad = '<pad>'
unk = '<unk>'
eos = '<eos>'
sos = '<sos>'
mask = '<mask>'


class WordVocab(object):
    def __init__(self, char_lst):
        super(WordVocab, self).__init__()
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        self.char_lst = char_lst
        self._char2idx = {
                        '<pad>': self.pad_index,
                        '<unk>': self.unk_index,
                        '<eos>': self.eos_index,
                        '<sos>': self.sos_index,
                        '<mask>': self.mask_index
                    }

        for char in self.char_lst:
            if char not in self._char2idx:
                self._char2idx[char] = len(self._char2idx)
        self._idx2char = dict((idx, char) for char, idx in self._char2idx.items())
        print(f'vocab size: {self.vocab_size}')


    def char2index(self, chars):
        if isinstance(chars, list):
            return [self._char2idx.get(char, self.unk_index) for char in chars]
        else:
            return self._char2idx.get(chars, self.unk_index)

    def index2char(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2char.get(i) for i in idxs]
        else:
            return self._idx2char.get(idxs)


    @property
    def vocab_size(self):
        return len(self._char2idx)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


def load_data(path):
    assert os.path.exists(path)

    char_set = set((_pad, unk, eos, sos, mask))
    char_set.add(_pad)
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            if line != '':
                chars = line.strip().split(hp.split_mark)
                for c in chars:
                    char_set.add(c)
        return char_set



def build():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--vocab_path", required=None, type=str)
    parser.add_argument("-o", "--output_path", default='./data/', type=str)
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    args = parser.parse_args()
    char_lst = load_data('../data/corpus_pre.txt')
    print(len(char_lst))
    symbols = char_lst
    vocab = WordVocab(symbols)
    vocab.save_vocab('../data/vocab.test')
    print("保存vocab成功！")

if __name__ == '__main__':
    build()













