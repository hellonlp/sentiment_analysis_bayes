# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:46:08 2019

@author: cm
"""

import os
import sys
pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pwd)


from bayes import predictionBayes




if __name__ =='__main__':
    ### 测试
    print(predictionBayes('我爱武汉'))
    


