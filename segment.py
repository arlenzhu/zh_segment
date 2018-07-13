# -*- coding: utf-8 -*-
""" ------------------------------------------------- 
File Name: segment
Description : 
Author : arlen
date：18-7-13
------------------------------------------------- """
from segment_model import bigru_crf_model
import re
import numpy as np
import utlis
import pickle
from keras.preprocessing.sequence import pad_sequences


class Segment(object):
    def __init__(self):
        self.model = bigru_crf_model()
        self.model.load_weights('./model_file/model.hdf5')
        self.max_len = utlis.max_len
        self.tokensize = pickle.load(open('./model_file/tokenizer.pkl', 'rb'))
        self.dlm = re.compile('[^\u4e00-\u9fa5a-z\s]')
        self.tags = {1: 'B', 2: 'M', 3: 'E', 4: 'S'}
        zy = {'be': 0.5, 'bm': 0.5, 'eb': 0.5, 'es': 0.5, 'me': 0.5, 'mm': 0.5, 'sb': 0.5, 'ss': 0.5}
        self.zy = {i: np.log(zy[i]) for i in zy.keys()}

    def cut_word(self, doc):
        sents = re.finditer(self.dlm, doc)
        j = 0
        result = []
        for i in sents:
            strings = doc[j:i.start()]
            result.extend(self.pred_word(strings))
            result.extend(doc[i.start(): i.end()])
            j = i.end()
        result.extend(self.pred_word(doc[j:]))
        return result

    def pred_word(self, strings):
        if strings:
            words = self.tokensize.texts_to_sequences([list(strings), ])
            words_pad = pad_sequences(words, maxlen=self.max_len)
            pred = self.model.predict(words_pad)[0][-len(words[0]):]
            # pred = [utlis.tags.get(np.argmax(i), 'X') for i in pred][-len(words[0]):]
            nodes = [dict(zip(['b', 'm', 'e', 's'], i[1:])) for i in pred]
            best_tags = self.viterbi(nodes)

            words = []
            for i in range(len(strings)):
                if best_tags[i] in ['s', 'b']:
                    words.append(strings[i])
                else:
                    words[-1] += strings[i]
            return words
        else:
            return []

    def viterbi(self, nodes):
        paths = {'b': nodes[0]['b'], 's': nodes[0]['s']}
        for l in range(1, len(nodes)):
            paths_ = paths.copy()
            paths = {}
            for i in nodes[l].keys():
                nows = {}
                for j in paths_.keys():
                    if j[-1] + i in self.zy.keys():
                        nows[j + i] = paths_[j] + nodes[l][i] + self.zy[j[-1] + i]
                k = max(nows.items(), key=lambda x: x[1])
                paths[k[0]] = k[1]
        return max(paths.items(), key=lambda x: x[1])[0]


if __name__ == "__main__":
    seg = Segment()
    print(seg.cut_word('结婚的和尚未结婚的'))
    print(seg.cut_word('严守一把手机关了'))
    print(seg.cut_word('代表北大的人大代表，代表人大的北大博士'))
    print(seg.cut_word('RNN的意思是，为了预测最后的结果，我先用第一个词预测，当然，只用第一个预测的预测结果肯定不精确，我把这个结果作为特征，跟第二词一起，来预测结果；接着，我用这个新的预测结果结合第三词，来作新的预测；然后重复这个过程。'))