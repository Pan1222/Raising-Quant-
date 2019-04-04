#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 18:23:47 2019

@author: peter
Binomial Tree for option pricing 
"""

import numpy as np
import math 
import matplotlib.pyplot as plt
from scipy import stats 

#define binomial tree class, which can plot the price evolution and payoff and also calculate option price
class BinomialTree:
    def __init__(self,spot,strike,riskfree,divyd,sigma,maturity,steps):
        self.p = spot
        self.s = strike
        self.r = riskfree
        self.d = divyd
        self.sigma = sigma
        self.deltatime = maturity/steps
        self.steps = steps
        self._calpara()
        self._buildtree()#once initialized, price evolution will be determined
        
    #calculate up,down p
    def _calpara(self):
        """
            calculate up down and p for binomial tree
        """
        self.up = math.exp(self.sigma*math.sqrt(self.deltatime))
        self.down = math.exp(-self.sigma*math.sqrt(self.deltatime))
        self.upprob = (math.exp((self.r-self.d)*self.deltatime)-self.down)/(self.up-self.down)
        
    #build the tree
    def _buildtree(self):
        """
            derive the price evolution tree
        """
        self.pricetree = np.zeros((self.steps+1,self.steps+1))
        self.pricetree[0][0] = self.p
        for j in range(self.steps):
            for i in range(j+1):
                self.pricetree[j+1][i+1] = self.pricetree[j][i]*self.down
            self.pricetree[j+1][0] = self.pricetree[j][0]*self.up
    
    #plot the final price distribution
    def plotprice(self):
        """
            plot the final price distribution 
        """
        plt.figure()
        plt.hist( self.pricetree[-1,:] )
        plt.title("price Distribution")    
        plt.show()
        
    #plot the final price distribution
    def plotpayoff(self):
        """
            plot the final payoff 
        """
        plt.figure()
        payoff = list(map(lambda x:max(x-self.s,0.0),self.pricetree[-1,:]))
        plt.plot(payoff)
        plt.title("Payoff Distribution")
        plt.show()
    
    #use funtion to return option price
    def priceit(self):
        """
            call this function to price the option
        """
        paytree = np.zeros((self.steps+1,self.steps+1))
        paytree[-1,:] = np.array( list( map(lambda x:max(x-self.s,0.0),self.pricetree[-1,:]) ) )
        discount = math.exp( self.r*self.deltatime )
        for i in range(self.steps,0,-1):
            for j in range(i):
                paytree[i-1][j] = (paytree[i][j]*self.upprob +paytree[i][j+1]*(1-self.upprob))/discount
        return paytree[0][0]
            
    
#继承美式期权-子类
class extendedtree(BinomialTree):
    #calculate american option price
    def americanprice(self):
        """
            call this function to price american option
        """
        self.americanpay = np.zeros((self.steps+1,self.steps+1))
        self.optionvalue = np.zeros((self.steps+1,self.steps+1))
        self.exercisevalue = np.zeros((self.steps+1,self.steps+1))
        self.americanpay[-1,:] = np.array( list( map(lambda x:max(x-self.s,0.0),self.pricetree[-1,:]) ) )
        discount = math.exp( self.r*self.deltatime )
        for i in range(self.steps,0,-1):
            for j in range(i):
                self.optionvalue[i-1][j] = (self.americanpay[i][j]*self.upprob + self.americanpay[i][j+1]*(1-self.upprob))/discount
                self.exercisevalue[i-1][j] = max(self.pricetree[i-1][j]-self.s,0.0)
                self.americanpay[i-1][j] = max(self.optionvalue[i-1][j],self.exercisevalue[i-1][j])
        return self.americanpay[0][0]


def BSM(spot,strike,riskfree,sigma,ttm):
    """
        call to calculate price based on BS formula
    """
    d1 = (math.log(spot/strike,math.e)+(riskfree+sigma**2/2)*ttm)/(sigma*math.sqrt(ttm))
    d2 = d1 - sigma*math.sqrt(ttm)
    call = stats.norm.cdf(d1)*spot - strike*math.exp(-riskfree*ttm)*stats.norm.cdf(d2)
    return call


#对于满足下列条件的call option价格进行收敛比较 
spot = 100
strike = 110
riskfree = 0.03
sigma = 0.2
maturity = 1
div = 0.02

option = BSM(spot,strike,riskfree,sigma,maturity)

priceprocess = []
priceprocess2 = []

for step in range(50,2000,150):
    tree = BinomialTree(spot,strike,riskfree,div,sigma,maturity,step)
    tree2 = extendedtree(spot,strike,riskfree,div,sigma,maturity,step)
    priceprocess.append(tree.priceit())
    priceprocess2.append(tree2.americanprice())
    
    
plt.figure()
plt.plot( range(50,2000,150),priceprocess,'b--' )
plt.plot( range(50,2000,150),priceprocess2,'r--' )
#plt.plot(range(50,2000,150),[option]*len(range(50,2000,150)),'g-')
plt.legend(['eurooption','americaoptionn','bsprice'])
plt.xlabel('num of steps')
plt.ylable('price')
plt.title('call option price')
plt.savefig(u'./二叉树价格收敛.pdf')
plt.show()


#result1: div=0时，美式价格和欧式一模一样




    





























    
    
    
    
    
    
    