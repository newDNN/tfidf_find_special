#coding:utf-8
import codecs
import json
import math
import os
import pickle
from collections import defaultdict, Counter
from tools import doc2list, load_stopwords


class ComTfIdf(object):
    '''
    计算tf(word, one doc)
    计算idf(word, docs集合)：判断该词在不在一个文档里：考虑到文档太多，从资源管理器中每次加载一个doc
    辅助函数doc2list(docfile，is_set):将doc每个句子分词去停用词形成列表，一个doc表示为一个很多句子concat后的长列表
    '''
    def __init__(self,doc_path):
        self.doc_path=os.path.abspath(os.path.join(os.path.curdir,doc_path))
        self.stopwords = load_stopwords()

    def get_all_words(self):
        word_set = set()
        file_list = os.listdir(self.doc_path)
        cnt = 0
        for i in file_list:
            cnt += 1
            path = os.path.join(self.doc_path, i)
            if os.path.isfile(path):
                doc_list = doc2list(path, is_set=True)
                word_set |= set(doc_list)
            if cnt%20==0:
                print('count words from {} docs'.format(cnt))
        word_list = list(word_set)
        return word_list


    def calc_idf(self):
        '''
        :param word_list:
        :return: {word:idf值}
        '''
        Cnt = defaultdict(int)
        file_list = os.listdir(self.doc_path)
        N = len(file_list)

        print('total docs:{}'.format(N))
        cnt = 0
        for i in file_list:
            cnt += 1
            if cnt % 20 == 0:
                print('calc {} docs idf'.format(cnt))
            path = os.path.join(self.doc_path, i)
            assert os.path.isfile(path)
            doc_list = doc2list(path, is_set=True)
            for word in doc_list:
                Cnt[word] += 1
            #print('doc words:{},all_words:{}'.format(len(doc_list),len(word_list)))
        Idf = {}
        for w in Cnt:
            idf = math.log(1.0 * N / (Cnt[w]+1))
            Idf[w]=idf
        return Idf

    @staticmethod
    def calc_tf(word, doc_list):
        tmp = list(set(doc_list))
        total_words = len(tmp)
        #print(doc_list)
        if word in doc_list:
            return doc_list.count(word)*1.0/total_words
        return 0.0

    def save_idf(self):
        Idf = self.calc_idf()
        with open('1994word_idf.json', 'w', encoding='utf-8') as f:
            json.dump(Idf, f, ensure_ascii=False)
        # pickle.dump(Idf, open(self.path+'word_idf.pkl','wb'))
        print('saved word_idf~~')

    # def save_allwords(self):
    #     with open('all_words_list.json', 'w', encoding='utf-8') as f:
    #         json.dump(self.words, f, ensure_ascii=False)
    #     # pickle.dump(self.words, open(path+'all_words_list.pkl','wb'))
    #     print('saved all words in hotel comments~~')



if __name__=='__main__':

    com = ComTfIdf(doc_path='1994sanya_hotels')

    com.save_idf()
    # com.save_allwords()

