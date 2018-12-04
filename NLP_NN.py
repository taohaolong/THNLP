# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import imdb
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing import sequence


def NLP_NN():
    ## EDA
    # 加载数据，这个数据来自： https://s3.amazonaws.com/text-datasets/imdb_full.pkl
    (x_train, y_train), (x_test, y_test) = imdb.load_data()
    # 探索一下数据情况
    lens = list(map(len, x_train))
    avg_len = np.mean(lens)
    print(avg_len)
    plt.hist(lens, bins=range(min(lens), max(lens) + 50, 50))
    plt.show()

    # 由于长度不同，这里取相同的长度
    m = max(max(list(map(len, x_train))), max(list(map(len, x_test))))
    print('m=%d' % m)
    maxword = min(400, m)
    x_train = sequence.pad_sequences(x_train, maxlen=maxword)
    x_test = sequence.pad_sequences(x_test, maxlen=maxword)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    #词数
    vocab_siz = np.max([np.max(x_train[i]) for i in range(x_train.shape[0])]) + 1
    print('vocab_siz=%d' % vocab_siz)
    print('x_train.shape=[%d,%d]' % (x_train.shape[0], x_train.shape[1]))
    #构建模型
    model = Sequential()  #序贯模型
    # 第一层是嵌入层,矩阵为 vocab_siz * 64
    model.add(Embedding(vocab_siz, 64, input_length=maxword))
    # 把矩阵压平，变成vocab_siz * 64维的向量
    model.add(Flatten())
    # 加入多层全连接
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(50, activation='relu'))
    # 最后一层输进0~1之间的值，像lr那样
    model.add(Dense(1, activation='sigmoid'))
    # 计算
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    print(type(x_train))
    #训练
    model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=100, nb_epoch=20, verbose=1)
    score = model.evaluate(x_test, y_test)
    print(score)

if __name__=='__main__':
    NLP_NN()