# coding:utf-8

import model

# 测试文件位置
testfile = 'data/train/poc_test.txt'

# SVM模型预测
a = model.SVM()
# LogisticRegression模型预测
# a = model.LG()

with open(testfile, 'r') as f:
    print('Testfile： ' + testfile)
    preicdtlist = [i.strip('\n') for i in f.readlines()[:]]
    result = a.predict(preicdtlist)
    print('First 10 Malicious Requests: ' + str(result[1][:10]))
    print('First 10 Normal Requests: ' + str(result[0][:10]))
    pass
