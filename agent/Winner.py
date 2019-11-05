#-*- coding:utf-8 -*-
'''
@Author: Louis Liang
@time:2018/9/15 9:16
'''

import numpy as np

class WINNER:
    def __init__(self):
        self.a_dim=0

    def predict(self,s,a):
        close=[]
        for i,prices in enumerate(s[0]):
            closes=prices[-1]
            close.append(closes[-1]/closes[-2])
        weights = np.zeros(len(s[0]))
        weights[np.argmax(close)] = 1
        weights = weights[None,:]
        return weights