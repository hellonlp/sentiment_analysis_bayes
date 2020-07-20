# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:49:16 2019

@author: cm
"""


import os
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)
import numpy as np
from bayes import train,read_vector



if __name__ =='__main__':
    # Load y
    labels = np.loadtxt(os.path.join(pwd,'data','types.txt'))
    # Load x
    #vectors = read_vector('vector_pearson_40000.txt')
    vectors = read_vector('vectors1000.txt')
    # Training
    p0Vec,p1Vec,pClass1 = train(vectors,labels)




















