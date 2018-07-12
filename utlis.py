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


dlm = "，。！？、；："
re_dlm = re.compile('[%s]' % dlm)


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
        return line

    d = list(map(trans_, d))

    return d


def buid_data(doc):
    x = []
    y = []
    for sent in doc:
        sents = re.split(re_dlm, sent)
        for line in sents:
            words = [i for i in line.split('  ') if i != '']
            if 1 < len(words) < 100:
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
                x.append(' '.join(l_x))
                y.append(' '.join(l_y))
    return x, y


if __name__ == '__main__':
    train_data = read_data('./train_data/msr_training.utf8')

    train_x, train_y = buid_data(train_data)

    corpus = train_x

    words = len(set(' '.join(corpus).split()))

    max_len = max([len(i) for i in corpus])

    tokenizer = Tokenizer(num_words=words)
    tokenizer.fit_on_texts(corpus)

    train_x = tokenizer.texts_to_sequences(train_x)
    train_x = pad_sequences(train_x, maxlen=max_len)

    tags = {'B': 1, 'M': 2, 'E': 3, 'S': 4}


    def pad_label(inlist):
        label_ls = []
        for i in inlist:
            label = [tags.get(j, 5) for j in i.split()]
            tmp = np.zeros(max_len)
            tmp[-len(label):] = label
            label_ls.append(tmp)
        return label_ls


    train_lable = np.expand_dims(np.array(pad_label(train_y)), 2)

    train_x.dump('./training/train_x.np')
    train_lable.dump('./training/train_y.np')