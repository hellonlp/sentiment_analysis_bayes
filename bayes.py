# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 13:40:26 2018

@author: cm
"""
import os
import jieba
import numpy as np


pwd = os.path.dirname(os.path.abspath(__file__))

#训练
def train(trainMatrix,trainCategory):
    """
    情感分为两类：正面和负面，1为正面，0为负面。
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num,p1Num = np.ones(numWords),np.ones(numWords)
    p0Deom,p1Deom = 2,2
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num = p1Num + trainMatrix[i]
            p1Deom = p1Deom + sum(trainMatrix[i])
        else:
            p0Num = p0Num + trainMatrix[i]
            p0Deom = p0Deom + sum(trainMatrix[i])  #向量相加
    p1Vect = p1Num/p1Deom      # 对每个元素做除法
    p0Vect = p0Num/p0Deom  
    p1VectLog = np.zeros(len(p1Vect))
    for i in range(len(p1Vect)):
        p1VectLog[i] = np.log(p1Vect[i])
    p0VectLog = np.zeros(len(p0Vect))        
    for i in range(len(p0Vect)):
        p0VectLog[i] = np.log(p0Vect[i])        
    return p0VectLog,p1VectLog,pAbusive    


# 朴素贝叶斯分类函数
def classify(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec) + np.log(pClass1)
    p0=sum(vec2Classify*p0Vec) + np.log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

#加载模型
def load_p0Vec_p1Vec_pClass1():
    f = os.path.join(pwd,'parametre_pearson_40000','p0Vec.txt')
    with open(f,encoding='utf-8') as fp:
        lines = fp.readlines()
    p0Vec = [float(l) for l in lines]
    p0Vec = np.array(p0Vec)
    #
    f = os.path.join(pwd,'parametre_pearson_40000','p1Vec.txt')
    with open(f,encoding='utf-8') as fp:
        lines = fp.readlines()
    p1Vec=[float(l) for l in lines]
    p1Vec = np.array(p1Vec)
    #
    f = os.path.join(pwd,'parametre_pearson_40000','pClass1.txt')
    with open(f,encoding='utf-8') as fp:
         lines=fp.readlines()
    pClass1 = float(lines[0])
    return p0Vec,p1Vec,pClass1



# 读取停用词(或者标点符号)
f = os.path.join(pwd,'dict','ponctuation_sentiment.txt')
with open(f,encoding='utf-8') as fp:
    lines = fp.readlines()
stopwords = set([line.strip('\n')  for line in lines])
  
  
# 去除停用词 ok
def drop_stopwords(sentence):
    segResult = jieba.lcut(sentence)
    newSent = []
    for word in segResult:
        if word in stopwords:
            continue
        else:
            newSent.append(word)
    return newSent


#读取词汇特征
f = os.path.join(pwd,'data','vocabulary_pearson_40000.txt')
with open(f,encoding='utf8') as fp:
    vocabulary = fp.readlines()
vocabulary = [texte.replace('\n','') for texte in vocabulary ]
def set_vector(sentence):
    line = drop_stopwords(sentence)
    vector = []
    for word in vocabulary:
        vector.append(int(line.count(word)))
    return vector

    
def read_vector(name):
    f = os.path.join(pwd,'data', name )
    with open(f) as fp:
        lines=fp.readlines()
    lines_new=[list(map(int,line.split())) for line in lines]
    return(np.array(lines_new))



#利用模型预测
p0Vec,p1Vec,pClass1 = load_p0Vec_p1Vec_pClass1()
def predictionBayes(Sentence):
    vector  = set_vector(Sentence)
    p = classify(vector,p0Vec,p1Vec,pClass1)
    if p == 1:
        return '正面'
    elif p == 0:
        return '负面'




if __name__ =='__main__':
    #--- 测试 ---#
    print(predictionBayes('我爱武汉'))
    
    
    
    
    
