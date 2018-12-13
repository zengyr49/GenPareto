# -*- coding: utf-8 -*-
# @Time    : 2018/7/24 17:03
# @Author  : ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : GenParetoOptFunc.py
# @Software: PyCharm

import numpy as np

class GenParetoOptFunc:
    __doc__ = "Changed formula to be optimize. To optimize G(theta) for calculating the shape and scale of " \
              "Generalized Pareto Distribution. This function is minimized by Gradient Descent or PSO." \
              "reference:'A new hybrid estimation method for the generalized pareto distribution' by Chunlin Wang & Gemai Chen"

    def __init__(self,X,theta,threshold=0):
        # X is the input data, no matter you sorted or pick the exceedance
        # theta is the parameter to be found to minimize the function
        # threshold is the threshold of exceedance
        NeedX = []
        for i in X:
            if i > threshold:
                exceedance = i - threshold
                NeedX.append(exceedance)
        self.X = np.array(sorted(NeedX))
        self.theta = theta

    def g(self,idx,sumpow):
        # idx is index of the data entry
        # sumpow is a part of formula
        xi = self.X[idx]
        sumfunc = 1 - self.theta * xi
        sumfunc1 = sumfunc ** sumpow
        result = 1 - sumfunc1
        return result

    def lenX(self):
        try:
            return self.X.shape[1]
        except:
            return len(self.X)

    def sumlogfunc(self):
        print(self.theta)
        sumfunc1 = np.sum(np.log(1 - self.theta * self.X))
        return sumfunc1

    def sumpowfunc(self):
        sumfunc1 = self.sumlogfunc()
        XLength = self.lenX()
        return -1 * XLength / sumfunc1

    def sumfunc_gprai_p3(self):
        # for calculation sum(X/(1-theta*X))
        return np.sum(self.X / (1 - self.theta * self.X))

    def gprai_p1(self,idx,sumpow):
        # for -(1-theta*xi)^sumpow
        xi = self.X[idx]
        return -1 * (1 - self.theta * xi) ** sumpow

    def gprai_p2(self,idx,sumlog):
        # sumlog is from sumlogfunc. It is sum(log(1-theta*X))
        xi = self.X[idx]
        XLength = self.lenX()
        a = XLength * xi
        b = 1 - self.theta * xi
        return a / (b * sumlog)

    def gprai_p3(self,idx,sumlog,sum_gprai_p3):
        xi = self.X[idx]
        XLength = self.lenX()
        a = np.log(1 - self.theta * xi)
        b = sum_gprai_p3
        c = sumlog
        return XLength * a * b / (c ** 2)

    def gprai(self,p1,p2,p3):
        return p1 * (p2 - p3)

    def Gpraisum_p(self,idx,sumlog,sumpow,sum_gprai_p3):
        n = self.lenX()
        gip1 = self.gprai_p1(idx=idx,sumpow=sumpow)
        gip2 = self.gprai_p2(idx=idx,sumlog=sumlog)
        gip3 = self.gprai_p3(idx=idx,sumlog=sumlog,sum_gprai_p3=sum_gprai_p3)
        giprai = self.gprai(p1=gip1,p2=gip2,p3=gip3)
        gi = self.g(idx=idx,sumpow=sumpow)
        one = (2 * (idx + 1) - 1) * giprai / gi
        two = (2 * n + 1 - 2 * (idx + 1)) * giprai / (1 - gi)
        return one - two

    def Gsum_p(self,idx,sumpow):
        n = self.lenX()
        gi = self.g(idx=idx,sumpow=sumpow)
        one = (2 * (idx + 1) - 1) * np.log(gi)
        two = (2 * n + 1 - 2 * (idx + 1)) * np.log(1 - gi)
        return one + two

    def Gprai(self):
        n = self.lenX()
        sumlog = self.sumlogfunc()
        sumpow = self.sumpowfunc()
        sum_gprai_p3 = self.sumfunc_gprai_p3()
        sum_Gp = 0
        for idx in range(n):
            # idxReal = idx + 1
            givalue = self.Gpraisum_p(idx=idx,sumlog=sumlog,
                                      sumpow=sumpow,sum_gprai_p3=sum_gprai_p3)
            sum_Gp = sum_Gp + givalue
        return -1 * sum_Gp / n

    def G(self):
        n = self.lenX()
        sumpow = self.sumpowfunc()
        sum_Gp = 0
        for idx in range(n):
            # idxReal = idx + 1
            givalue = self.Gsum_p(idx=idx,sumpow=sumpow)
            sum_Gp = sum_Gp + givalue
        return (-1 * n - sum_Gp) / n








