
# -*- coding:utf-8-*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt
# 分词
from pprint import pprint

import jieba
from bs4 import BeautifulSoup
from gensim import corpora
from keras.layers import Embedding, LSTM, Dense, Activation, Dropout
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.cross_validation import train_test_split


def cutPhase(inFile, outFile):
    # 如果没有自己定义的词典，这行不要
    # jieba.load_userdict("dict_all.txt")
    # 加载停用词
    stoplist = {}.fromkeys([line.strip() for line in open('data\\stopword.txt', 'r', encoding='utf-8')])
    f1 = open(inFile, 'r', encoding='utf-8')
    f2 = open(outFile, 'w+', encoding='utf-8')
    line = f1.readline()
    count = 0
    while line:
        b = BeautifulSoup(line, "lxml")
        line = b.text
        # 分词
        segs = jieba.cut(line, cut_all=False)
        # 过滤停用词
        segs = [word for word in list(segs)
                if word.lstrip() is not None
                and word.lstrip() not in stoplist
                ]
        # 每个词用空格隔开
        f2.write(" ".join(segs))
        f2.write('\n')
        line = f1.readline()
        count += 1
        if count % 100 == 0:
            print(count)
    f1.close()
    f2.close()


def load_data(out_pos_name='data/pos.txt', out_neg_name='data/neg.txt'):
    def do_load(file_name, dir):
        c = 0
        with open(file_name, 'w+', encoding='utf-8') as f_out:
            for root, _, files in os.walk(dir):
                # print(root)
                for f_name in files:
                    p = os.path.join(root, f_name)
                    try:
                        with open(p, mode='r', encoding='gbk') as f_read:
                            # print(os.path.join(root, f_name))
                            c += 1
                            txt = f_read.read()
                            txt = re.subn(r'\s+', ' ', txt)[0]
                            f_out.write('%s\n' % (txt))
                            # if c % 100 == 0:
                            #     print(c)
                    except Exception as e:
                        print('p:', p)
                        # print('e:',e)

    print('加载pos!!!')
    do_load(out_pos_name,
            'data/ChnSentiCorp_htl_ba_2000/pos')
    print('加载neg!!!')
    do_load(out_neg_name,
            'data/ChnSentiCorp_htl_ba_2000/neg')


def combine_data():
    c = 0
    f_w = open('data/train.cut', 'w+', encoding='utf-8')
    f_pos = open('data/pos.cut', 'r', encoding='utf-8')
    line = f_pos.readline()
    while line:
        c += 1
        f_w.write('%d\t%s' % (1, line))
        line = f_pos.readline()
        print(c)
    f_pos.close()

    f_neg = open('data/neg.cut', 'r', encoding='utf-8')
    line = f_neg.readline()
    while line:
        c += 1
        f_w.write('%d\t%s' % (0, line))
        line = f_neg.readline()
        print(c)
    f_neg.close()

    f_w.close()


if __name__ == '__main__':
    # print('# 加载数据')
    # load_data(out_pos_name='data/pos.txt', out_neg_name='data/neg.txt')
    # print('# 分词')
    # cutPhase(inFile='data/pos.txt', outFile='data/pos.cut')
    # cutPhase(inFile='data/neg.txt', outFile='data/neg.cut')
    # 数据融合
    # combine_data()
    Y = []
    x = []
    for line in open('data/train.cut', encoding='utf-8'):
        label, sentence = line.split("\t")
        Y.append(int(label))
        x.append(sentence.split())

    print('#构建字典')
    dic = corpora.Dictionary(x)
    X = []
    for row in x:
        tmp = []
        for w_i in row:
            tmp.append(dic.token2id[w_i])
        X.append(tmp)
    X = np.array(X)
    Y = np.array(Y)
    # lens = list(map(len, X))
    # avg_len = np.mean(lens)
    # print(avg_len)
    # plt.hist(lens, bins=range(min(lens), max(lens) + 50, 50))
    # plt.show()

    # 由于长度不同，这里取相同的长度,平均长度为38.18，最大长度为337.
    m = max(list(map(len, X)))
    print('m=%d' % m)
    maxword = min(100, m)
    X = sequence.pad_sequences(X, maxlen=maxword)
    print(X.shape)

    ## 数据划分
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # 构建模型
    model = Sequential()
    model.add(Embedding(len(dic) + 1, 128, input_length=maxword))
    # model.add(LSTM(128, dropout_W=0.2, return_sequences=True))
    # model.add(LSTM(64, dropout_W=0.2,return_sequences=True))
    model.add(LSTM(128, dropout_W=0.2))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))
    # 计算
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(model.summary())
    # 进行训练
    model.fit(x_train, y_train, batch_size=100, nb_epoch=10, validation_data=(x_test, y_test))
    ## 结果评估
    score, acc = model.evaluate(x_test, y_test, batch_size=100)
    print("score: %.3f, accuracy: %.3f" % (score, acc))

    # # 预测
    # my_sentences = ['讨厌 房间']
    # my_x = []
    # for s in my_sentences:
    #     words = s.split()
    #     tmp = []
    #     for w_j in words:
    #         tmp.append(dic.token2id[w_j])
    #     my_x.append(tmp)
    # my_X = np.array(my_x)
    # my_X = sequence.pad_sequences(my_X, maxlen=maxword)
    # labels = [int(round(x[0])) for x in model.predict(my_X)]
    # for i in range(len(my_sentences)):
    #     print('%s:%s' % ('正面' if labels[i] == 1 else '负面', my_sentences[i]))

