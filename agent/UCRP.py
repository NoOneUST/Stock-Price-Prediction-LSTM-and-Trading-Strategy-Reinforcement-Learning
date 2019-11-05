#-*- coding:utf-8 -*-
'''
@Author: Louis Liang
@time:2018/9/15 9:13
'''
import numpy as np

class UCRP:
    def __init__(self):
        self.a_dim=0

    def predict(self,s,a):
        weights=np.ones(len(a[0]))/len(a[0])
        weights=weights[None,:]
        return weights