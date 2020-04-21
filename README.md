# MaliciousUrls_Part2

### 基于机器学习的恶意Url检测第二版

  基于SVM和逻辑回归算法，结合IF-TDF模型对数据进行预处理，机上kmeans算法初步进行特征提取，实现恶意url检测。
 
 ### 模型训练+预测
 
  运行start.py可自动训练模型并预测制定的文件，可在model.py中指定kmeans聚合维度和ngram分词法的格式。
  
 ### 数据格式
 
  数据存放文件夹默认为白样本存放在data/good中，黑样本存放在data/bad中，测试文件可通过start.py指定位置。
 
 ### 数据样式
 
  url样式参照样本文件，同时提供pcap.py实现从pcap包中自动提取http包的url。
  
 ### 运行环境
 
  python 3.7
 
 ### 依赖包
 
  sklearn、scapy_http
