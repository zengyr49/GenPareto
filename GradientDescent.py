# -*- coding: utf-8 -*-
# @Time    : 2018/7/24 19:02
# @Author  : ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : GradientDescent.py
# @Software: PyCharm
import numpy as np
import random
import GenParetoOptFunc

def derivative(X,theta,threshold=0):
    gpof = GenParetoOptFunc.GenParetoOptFunc(X,theta,threshold)
    return gpof.Gprai()

def target_func(X,theta,threshold=0):
    gpof = GenParetoOptFunc.GenParetoOptFunc(X, theta, threshold)
    return gpof.G()

def GradientDescent(X,step=0.0001,acc=10e-06,maxIter=1000,showdetail=False,threshold=0):
    # step can be interpreted as learning rate, and lr will change according to iterations
    # acc is acception or precision
    X = sorted(X)
    curX = random.uniform(0,1 / max(X))
    foreX = curX
    iterCount = 1
    previousGradient = 0
    while True:
        lr = step / (np.sqrt(previousGradient + acc))
        gradient = derivative(X=X,theta=curX,threshold=threshold)
        curX = curX - lr * gradient
        diff = abs(curX - foreX)
        foreX = curX
        funcVal = target_func(X=X,theta=curX,threshold=threshold)
        if showdetail == True:
            print("Optimized at Func =",funcVal)

        iterCount = iterCount + 1
        if iterCount >= maxIter or diff <= acc:
            optimizedX = curX
            optimizedVal = funcVal
            if showdetail == True:
                print("Optimization converge at Func =",funcVal)
            break
    return optimizedX
