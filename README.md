# MaliciousUrls2 - 机器学习检测恶意url第二版

<p align="center">
    <a><img src="https://raw.githubusercontent.com/Coldwave96/MaliciousUrls_Part2/master/MaliciousUrls2.png"/></a>
</p>

<p align="center">
    <a><img src="https://img.shields.io/badge/Python-3.7+-blue"></a>
    <a><img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-orange"></a>
    <a><img src="https://img.shields.io/github/license/coldwave96/MaliciousUrls_Part2"></a>
    <a><img src="https://img.shields.io/github/v/release/coldwave96/MaliciousUrls_Part2"></a>
</p>

<p align="center">
    <a href="https://coldwave96.github.io/">Welcome to my personal blog（＾◇＾）</a>
</p>

<hr>

## 介绍

- [√] 基于机器学习的恶意Url检测第二版

- [√] 通过IF-TDF模型对数据进行预处理

- [√] kmeans算法初步进行特征提取

- [√] SVM和逻辑回归算法建模
 
## 使用
 
* git clone 项目 or 下载 Release

* cd 项目文件夹 && `pip3 install -r requirements.txt`

* 待检测url在start.py中设置

* `python3 start.py`即可训练模型并自动预测，打包的项目文件里已经有训练好的模型，可以直接运行

* 可以参照下面的[说明](#maliciousurls2---url)设置样本和模型参数重新训练模型

## 说明

### 模型训练

* 可在model.py中指定kmeans聚合维度和ngram分词法的格式
  
### 数据格式

* 数据存放文件夹默认为白样本存放在/data/good中，黑样本存放在/data/bad中，测试文件可通过start.py指定位置。
 
### 数据样式
 
* url样式参照样本文件，同时提供pcap.py实现从pcap包中自动提取http包的url。

* 在pcap.py中指定pcap文件，执行`python3 pcap.py`即可
