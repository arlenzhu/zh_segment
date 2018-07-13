# -*- coding: utf-8 -*-
""" ------------------------------------------------- 
File Name: utlis
Description : 
Author : arlen
date：18-7-11
------------------------------------------------- """
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle


re_dlm = re.compile('[^\u4e00-\u9fa5a-z\s]')
tags = {'B': 1, 'M': 2, 'E': 3, 'S': 4}
max_len = 50


def read_data(filepath):
    with open(filepath, encoding='utf-8') as f:
        d = f.read().split('\n')

    def trans_(line):
        if '“' not in line:
            line = line.replace('”', '')
        elif '”' not in line:
            line = line.replace('“', '')
        elif '‘' not in line:
            line = line.replace('’', '')
        elif '’' not in line:
            line = line.replace('‘', '')
        return line.lower()

    d = list(map(trans_, d))

    return d


def buid_data(doc):
    x = []
    y = []
    for sent in doc:
        sents = re.split(re_dlm, sent)
        for line in sents:
            words = [i for i in line.split('  ') if i != '']
            l_x = []
            l_y = []
            for word in words:
                l_x.extend(list(word))
                if len(word) == 1:
                    l_y.extend('S')
                elif len(word) == 2:
                    l_y.extend(['B', 'E'])
                elif len(word) > 2:
                    tmp = ['M'] * len(word)
                    tmp[0] = 'B'
                    tmp[-1] = 'E'
                    l_y.extend(tmp)
            if 1 <= len(l_x) <= max_len:
                x.append(' '.join(l_x))
                y.append(' '.join(l_y))
    return x, y


def load_data():
    train_data = read_data('./train_data/msr_training.utf8')

    train_x, train_y = buid_data(train_data)

    words = len(set(' '.join(train_x).split()))

    tokenizer = Tokenizer(num_words=words)
    tokenizer.fit_on_texts(train_x)

    train_seq = tokenizer.texts_to_sequences(train_x)
    train_seq = pad_sequences(train_seq, maxlen=max_len)

    def pad_label(inlist):
        label_ls = []
        for i in inlist:
            tag = i.split()
            tmp = np.zeros((len(tag), 5))
            for idx, k in enumerate(tag):
                tmp[idx][tags.get(k, 0)] = 1

            label2 = np.array([[1, 0, 0, 0, 0]] * max_len)
            label2[-len(tmp):] = tmp
            label_ls.append(label2)
        return np.array(label_ls)

    train_lable = pad_label(train_y)

    train_seq.dump('./training/train_x.np')
    train_lable.dump('./training/train_y.np')

    data_infor = {'max_len': max_len, 'num_words': words}

    pickle.dump(tokenizer, open('./model_file/tokenizer.pkl', 'wb'))
    pickle.dump(data_infor, open('./model_file/data_infor.pkl', 'wb'))


if __name__ == "__main__":
    load_data()
