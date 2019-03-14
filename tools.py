#coding:utf-8
import codecs
import pickle

import jieba
import numpy as np


file_stopwords = 'stopwords.txt'


def load_stopwords():
    with codecs.open(file_stopwords, 'r', encoding='utf-8') as f:
        data = f.readlines()
        stopword_list = [w.strip() for w in data]
    return stopword_list


def doc2list(doc_file, is_set):
    stopwords = load_stopwords()
    with codecs.open(doc_file, encoding='utf-8') as f:
        try:
            lines = f.readlines()
        except:
            print(doc_file)

        sentences = [line.strip() for line in lines]
        sen_list = [[w for w in list(jieba.cut(sentence)) if w not in stopwords] for sentence in sentences]
        if not sen_list:
            return []
        doc_list = list(np.concatenate(sen_list))
        if is_set:
            doc_list = list(set(doc_list))
    return doc_list


def load_idf(file):
    import json
    return json.load(open(file, encoding='utf-8'))
    # return pickle.load(open(dirpath+'word_idf.pkl', 'rb'))
