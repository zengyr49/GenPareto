# -*- coding: utf-8 -*-
# @Time    : 2018/7/24 19:54
# @Author  : ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : PSOptim.py
# @Software: PyCharm

from pyswarm import pso
import GenParetoOptFunc
import numpy as np

def OptByPSO(X,threshold=0):
    X = sorted(X)
    # here is the target function to be optimized by pso
    def TargetFunc(theta):
        Gtheta = GenParetoOptFunc.GenParetoOptFunc(X=X,theta=theta,threshold=threshold).G()
        return Gtheta

    xopt,fopt = pso(TargetFunc,lb=[-1000],ub=[1 / max(X)])
    return xopt[0]

if __name__ == '__main__':
    x = [1.7, 2.2, 14.4, 1.1, 0.4, 20.6,
       5.3, 0.7, 1.9, 13, 12, 9.3, 1.4,
       18.7, 8.5, 25.5, 11.6, 14.1, 22.1,
       1.1, 2.5, 14.4, 1.7, 37.6, 0.6, 2.2,
       39, 0.3, 15, 11, 7.3, 22.9, 1.7, 0.1,
       1.1, 0.6, 9, 1.7, 7, 20.1, 0.4, 2.8,
       14.1, 9.9, 10.4, 10.7, 30, 3.6, 5.6,
       30.8, 13.3, 4.2, 25.5, 3.4, 11.9, 21.5,
       27.6, 36.4, 2.7, 64, 1.5, 2.5, 27.4, 1,
       27.1, 20.2, 16.8, 5.3, 9.7, 27.5, 2.5, 27]
    theta = OptByPSO(x,0)
    # print(theta)




