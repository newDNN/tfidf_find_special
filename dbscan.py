#coding:utf-8
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tools.tokenizer.wordCut import WordCut
from tools.labelMap.labelText import LabelText
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import numpy as np
from w2c_cluster import sen2vector, cos,most_similar
from functools import reduce
from collections import defaultdict
import numpy as np



def tfidf(corpus_file):
    with open(corpus_file, encoding='utf8') as f:
        lines = f.readlines()
        corpus = [line.strip() for line in lines]
    tf_idf = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b")
    weight = tf_idf.fit_transform(corpus).toarray()
    np.save('weight.npy', weight)
    print('----save weight done--------')
    return weight

def embedding(corpus_file):
    data = []
    with open(corpus_file, encoding='utf8') as f:
        lines = f.readlines()
        corpus = [line.strip() for line in lines]
        data=[sen2vector(t) for t in corpus]
    return data

def cos_similar(data):
    res = defaultdict(list)
    done = []
    length = len(data)
    for i in range(length):
        if i in set(reduce(lambda x,y:x+y, res.values(), list(res.keys()))):
            continue
        for j in range(i+1,length):
            if (i,j) in done:
                continue
            score = cos(data[i], data[j])
            done.append((i,j))
            if score and score > 0.3:
                res[i].append(j)

    return res

def cos_similar2(corpus_file,save_path):
    o = open(save_path, 'w', encoding='UTF-8')
    with open(corpus_file, encoding='utf8') as f:
        lines = f.readlines()
        corpus = [line.strip() for line in lines]
        cnt = 0
        for i in corpus:
            cnt += 1
            corpus.remove(i)
            score_dict = {}
            for j in corpus:
                try:
                    score = cos(sen2vector(i), sen2vector(j))
                    if score and score > 0.4:
                        score_dict[j] = score
                        corpus.remove(j)
                except Exception as e:
                    print(repr(e))

            score_list = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
            o.write(str(cnt)+'\t'+i+'\n')
            for item in score_list:
                o.write(str(cnt)+'\t'+item[0]+'\n')
    o.close()

def cluster(corpus_file, method='dbscan', presentation='tfidf'):
    weight = eval(presentation)(corpus_file)
    if method=='dbscan':
        db = DBSCAN(eps=0.001, min_samples=3).fit(weight)
    # print(db.core_sample_indices_)
        print('\n\n',db.labels_)
        print('----------cluster done-------------')
        return db.labels_
    elif method=='kmeans':
        kmeans = KMeans(n_clusters=35)
        lab = kmeans.fit(weight)
        return lab.labels_
    elif method=='cos':
        if presentation=='embedding':
            data = weight
        else:
            data = embedding(corpus_file)
        res_dict = cos_similar(data)
        labels = np.zeros(len(data))
        for i, key in enumerate(res_dict.keys(),start=1):
            idx = [key]+res_dict[key]
            labels[idx]=i
        return labels










def show(corpus_file, save_path, method, presentation):
    label = cluster(corpus_file, method,presentation)
    label_text = LabelText(label, corpus_file)
    label_text.sortByLabel(write_path=save_path,show=False,write=True)



# #
# f=open('clean_comments.txt', encoding='utf-8').readlines()
# o = open('seg.txt', 'w', encoding='utf-8')
# for line in f:
#     if len(line.split())<2:continue
#     o.write(line.split()[1]+'\n')
# o.close()


corpus_file = 'ticket2mining.txt'
save_path =  'sortedLabelText.csv'
# show(save_path, method='dbscan',presentation='tfidf')
# show(corpus_file, save_path, method='dbscan', presentation='embedding')
# show(corpus_file, save_path, method='cos', presentation='embedding')
# most_similar()

cos_similar2(corpus_file, save_path='parser_cluster.txt')