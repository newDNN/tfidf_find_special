#coding:utf-8
import codecs
import math
import pickle
import re
import os
import time

import jieba

from tools import doc2list, load_idf, load_stopwords
from tfidf import ComTfIdf
from collections import defaultdict, Counter
from sys import argv

# jieba.load_userdict('userdict.txt')

Idf = load_idf('1994word_idf.json')
path = 'docs/'
stopwords = load_stopwords()

#加载酒店id和酒店name的对应关系
id_name = open('sanya_id_name.txt', encoding='utf-8').readlines()
id_dict = dict()
for line in id_name:
    line = line.strip().split('\t')
    id_dict[line[0]] = line[1]

def count_tf(docpath):
    words_tf = defaultdict(int)
    # words_tf = Counter()
    flist = os.listdir(docpath)
    for id in flist:
        path = os.path.join(docpath,id)
        words_list = open(path, encoding='utf-8').readlines()
        words = [word for line in words_list for word in jieba.cut(line.strip()) if word not in stopwords and len(word)>1]
        for w in words:
            words_tf[w]+=1
        # words = Counter(words) #比较慢
        # words_tf.update(words)

    print('count {} files.'.format(len(flist)))
    res = sorted(words_tf.items(), key=lambda x:x[1], reverse=True)
    with open('all_words_tf_rank.txt','w', encoding='utf8') as o:
        for w,c in res:
            o.write('\t'.join([w, str(c)])+'\n')


def find_sp_sentence(docfile, save_file, topn):
    print('saving:'+docfile)
    tf_dict = defaultdict(float)
    doc_list = doc2list(docfile, is_set=False)
    tmp = list(set(doc_list))
    total_words = len(tmp)
    for w in tmp: #事先算出tf
        tf_dict[w] = doc_list.count(w) * 1.0 / total_words
    # print(doc_list)

    N=1994
    dft = math.log(1.0 * N)

    out = dict()
    cnt = 0
    with codecs.open(docfile, encoding='utf-8') as f:
        lines = f.readlines()
        sentences = [line.strip() for line in lines]
        sentences = list(set(sentences))

        # -----词语的tfidf
        # words = set(doc_list)
        # for w in words:
        #     tf = tf_dict.get(w)
        #     idf = Idf.get(w,dft)
        #     out[w] = tf*idf

        #--------短句的tfidf
        for comment in sentences:
            comment = re.split(',|，|\.|。|;|；|!|！|\?|？|……|~|、| ', comment)
            for short_sen in comment:
                if len(short_sen)>150:continue
                cnt += 1
                short = [w for w in list(jieba.cut(short_sen)) if w not in stopwords]
                if len(short) ==0:
                    continue
                tfidf_sen = 0.0
                for w in short:
                    tf = tf_dict[w]
                    idf = Idf.get(w,dft)
                    tfidf_sen += tf*idf
                out[short_sen] = tfidf_sen/len(short)


    # print('saved tfidf scores for {} short sentences'.format(cnt))
    sp_sen = codecs.open(save_file, 'w', encoding='utf-8')
    sorted_list = sorted(out.items(), key=lambda d: d[1], reverse=True)
    top = min(topn, len(sorted_list))
    for item in range(top):
        sp_sen.write(sorted_list[item][0] + '\t' + str(sorted_list[item][1]) + '\n')

    sp_sen.close()
    print('find {} sp sentences done~'.format(save_file))



if __name__ == '__main__':

    find_sp_sentence('原文档.txt', '对文档拆句并按tfidf值排序后的结果.txt', topn=10000000)

