#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:38:51 2019

@author: peter
content: simulate ito process and plot 
"""



import numpy as np
import math
from matplotlib import pyplot as plt

class stockprice:
    def __init__(self,spot,ttm,ret,sigma,steps):
        self.s0 = spot
        self.t = ttm
        self.u = ret
        self.sigma = sigma
        self.steps = steps
        self.delta_t = ttm/steps
        self._itoprocess() #调用私有方法时，不加self则不能准确找到命名空间
        
    def _itoprocess(self):
        self.series = np.zeros(self.steps+1)
        self.series[0] = self.s0
        drift = self.u*self.delta_t 
        var = self.sigma*math.sqrt(self.delta_t)
        for i in range(self.steps):
            s = np.random.standard_normal()
            self.series[i+1] = self.series[i]*(1+drift+var*s)

price = np.zeros((100+1,10))  
for i in range(10):
    price1 = stockprice(100, 1, 0.1, 0.2, 100)
    price[:,i] = price1.series

plt.figure()
plt.plot(price)
plt.xlabel('step')
plt.ylabel('stock price')
plt.title('price simulation')
plt.savefig('./itoprocess.pdf')
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
