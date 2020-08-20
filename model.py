# coding:utf-8

import os
import time
import urllib
import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from scipy.sparse import csr_matrix, lil_matrix, coo_matrix

# 定义样本文件位置
good_dir = 'data/good'
bad_dir = 'data/bad'

# kmeans聚合的维度
k = 80
# ngram系数
n = 2

# 是否使用kmeans
use_k = True

# 新定义输出方法，便于调试
def printT(word):
    a = time.strftime('%Y-%m-%d %H:%M:%S: ', time.localtime(time.time()))
    print(a + str(word))

# 读取数据
def getdata(filepath):
    with open(filepath, 'r') as f:
        data = [i.strip('\n') for i in f.readlines()[:]]
    return data

# 遍历文件夹中文件以读取数据
def load_files(dir):
    data = []
    g = os.walk(dir)
    for path, dirs, files in g:
        for filname in files:
            fulpath = os.path.join(path, filname)
            printT("load file: " + fulpath)
            t = getdata(fulpath)
            data.extend(t)
    return data

# 训练模型基类
class Baseframe(object):

    def __init__(self):
        pass

    # 训练
    def Train(self):

        # 读取数据
        printT("Loading Good Data:")
        good_query_list = load_files(good_dir)

        printT("Loading Bad Data:")
        bad_query_list = load_files(bad_dir)

        # 整合数据
        data = [good_query_list, bad_query_list]
        printT("Done, Good Numbers:" + str(len(data[0])) + " Bad Numbers:" + str(len(data[1])))

        # 打标记
        good_y = [0 for i in range(len(data[0]))]
        bad_y = [1 for i in range(len(data[1]))]

        y = good_y + bad_y

        # 数据向量化预处理
        # 定义矢量化实例
        self.vectorizer = TfidfVectorizer(tokenizer=self.get_ngrams)
        # 把不规律的文本字符串列表转换成规律的([i,j],weight)的矩阵X[url条数，分词总类的总数，理论上少于256^n]
        # i表示第几条url，j对应于term编号(或者说是词片编号）
        X = self.vectorizer.fit_transform(data[0] + data[1])
        printT("Data Dimentions： " + str(X.shape))

        # 通过kmeans降维
        if use_k:
            X = self.transform(self.kmeans(X))
            printT("Kmeans Succeed")

        printT("Devide Training Data")
        # 使用train_test_split分割X，y列表（testsize表示测试占的比例）（random为种子）
        # X_train矩阵的数目对应 y_train列表的数目(一一对应)  -->> 用来训练模型
        # X_test矩阵的数目对应(一一对应) -->> 用来测试模型的准确性
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        printT('Devide Succeed')
        printT('Begin Training:')
        printT(self.classifier)
        self.classifier.fit(X_train, y_train)

        # 使用测试值对模型的准确度进行计算
        printT(self.getname() + 'Model Accuracy:{}'.format(self.classifier.score(X_test, y_test)))

        # 保存训练结果
        with open('model/' + self.getname() + '.pickle', 'wb') as output:
            pickle.dump(self, output)

    # 数据预处理裁剪字符格式
    def get_ngrams(self, query):
        tempQuery = str(query)
        ngrams = []
        for i in range(0, len(tempQuery)-n):
            ngrams.append(tempQuery[i:i+n])
        return ngrams

    def kmeans(self, weight):
        printT('Matrix before kmeans： ' + str(weight.shape))
        weight = weight.tolil().transpose()
        # 同一组数据 同一个k值的聚类结果是一样的。保存结果避免重复运算
        try:
            with open('model/k' + str(k) + '.label', 'r') as input:
                printT('loading kmeans success')
                a = input.read().split(' ')

                self.label = [int(i) for i in a[:-1]]
        except FileNotFoundError:
            printT('Start Kmeans: ')

            clf = KMeans(n_clusters=k, precompute_distances=False)

            s = clf.fit(weight)
            printT(s)

            # 保存聚类的结果
            self.label = clf.labels_

            with open('model/k' + str(k) + '.label', 'w') as output:
                for i in self.label:
                    output.write(str(i) + ' ')
        printT('kmeans succeed,total: ' + str(k) + ' classes')
        return weight

    # 转换成聚类后结果输入转置后的矩阵返回转置好的矩阵
    def transform(self, weight):
        a = set()
        # 用coo存可以存储重复位置的元素
        row = []
        col = []
        data = []
        # i代表旧矩阵行号label[i]代表新矩阵的行号
        for i in range(len(self.label)):
            if self.label[i] in a:
                continue
            a.add(self.label[i])
            for j in range(i, len(self.label)):
                if self.label[j] == self.label[i]:
                    temp = weight[j].rows[0]
                    col += temp
                    temp = [self.label[i] for t in range(len(temp))]
                    row += temp
                    data += weight[j].data[0]

        newWeight = coo_matrix((data, (row, col)), shape=(k,weight.shape[1]))
        return newWeight.transpose()

    # 对新的请求列表进行预测
    def predict(self, new_queries):
        try:
            with open('model/' + self.getname() + '.pickle', 'rb') as input:
                self = pickle.load(input)
            printT('loading ' + self.getname() + ' model success')
        except FileNotFoundError:
            printT('start to train the ' + self.getname() + ' model')
            self.Train()
        printT('start predict:')
        # 解码
        new_queries = [urllib.parse.unquote(url) for url in new_queries]
        X_predict = self.vectorizer.transform(new_queries)

        if use_k:
            printT('Transform Data')
            X_predict = self.transform(X_predict.tolil().transpose())

        printT('Transform Succeed, Start Predicting:')
        res = self.classifier.predict(X_predict)
        printT('Predict Succeed, Total：' + str(len(res)))
        result = {}

        result[0] = []
        result[1] = []

        # 两个列表并入一个元组列表
        for q, r in zip(new_queries, res):
            result[r].append(q)

        printT('good query: ' + str(len(result[0])))
        printT('bad query: ' + str(len(result[1])))

        return result


class SVM(Baseframe):

    def getname(self):
        if use_k:
            return 'SVM__n'+str(n)+'_k'+str(k)
        return 'SVM_n'+str(n)

    def __init__(self):
        # 定理逻辑回归方法模型
        self.classifier = svm.SVC()


class LG(Baseframe):

    def getname(self):
        if use_k:
            return 'LG__n'+str(n)+'_k'+str(k)
        return 'LG_n'+str(n)

    def __init__(self):
        # 定理逻辑回归方法模型
        self.classifier = LogisticRegression()
