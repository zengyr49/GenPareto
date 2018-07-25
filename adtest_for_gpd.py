# -*- coding: utf-8 -*-
# @Time    : 2018/7/24 19:23
# @Author  : ZENG Yanru
# @Email   : 595438103@qq.com
# @File    : adtest_for_gpd.py
# @Software: PyCharm


import scipy.stats as ss
import math
import GradientDescent
import numpy as np
import PSOptim

class adtest_for_gpd:
    __doc__ = "this class is for calculating Anderson Darling Test for General Pareto Distribution"

    def __init__(self,X,x0,pValThres=0.05,OptMethod="pso"):
        self.X = np.array(sorted(X))
        self.x0 = x0
        self.pValThres = pValThres
        self.OptMethod = OptMethod

    def lenX(self):
        try:
            return self.X.shape[1]
        except:
            return len(self.X)

    def est_k_s(self,threshold):
        #0 idx is shape, or -k
        #2 idx is scale, or a or sigma
        if self.OptMethod == "gd":
            # version of GradientDescent
            theta = GradientDescent.GradientDescent(X=self.X,step=0.0001,acc=10e-06,maxIter=1000,showdetail=False,threshold=threshold)
        elif self.OptMethod == "pso":
            # version of PSO
            theta = PSOptim.OptByPSO(X=self.X,threshold=threshold) # if we use this, we need to install pyswarm package or put its code in the same filedir
        shape = np.sum(np.log(1 - theta * self.X)) / self.lenX()
        scale = -1 * shape / theta
        return (shape,scale)

    def Pareto(self,threshold):
        estks = self.est_k_s(threshold)
        k = float(-1 * estks[0])
        scale = float(estks[1])
        z = float(self.x0 - threshold)
        # print("x0 is:",self.x0)
        # print("k,z,scale is",k,z,scale)
        if k == 0:
            a = -1 * z / scale
            return 1 - math.exp(a)
        else:
            a = 1 - k * z / scale
            b = 1.0/k
            return 1-math.pow(a, b)

    def find_thres(self,itertime):
        idx = math.floor((itertime*0.01)*len(self.X))
        if idx >= len(self.X):
            onemore = self.X[idx]
        else:
            onemore = self.X[idx+1]
        return (self.X[idx]+onemore)/2

    def p_is_reach(self,threshold):
        z_list = self.adtest_makeZ(threshold)
        Asqr = self.adtest_Asqr(z_list)
        return self.p_val_record(Asqr)

    def adtest_makeZ(self,threshold):
        ordered_X = []
        for x in self.X:
            if x > threshold:
                ordered_X.append(x)
        ordered_X = sorted(ordered_X)
        estks = self.est_k_s(threshold)
        k = -1*estks[0]
        scale = estks[1]
        c_output = []
        for x in ordered_X:
            z = x - threshold
            if k == 0:
                x1 = self.k_is_zero(z,scale)
            else:
                x1 = self.k_is_not_zero(z,scale,k)
            c_output.append(x1)
        c_output_new = []
        for i in c_output:
            c_output_new.append(i)
        return c_output_new

    def k_is_zero(self,z,scale):
        return 1-math.exp(-z/scale)
    def k_is_not_zero(self,z,scale,k):
        # print("z,scale,k",z,scale,k)
        return 1-math.pow((1-k*z/scale),(1/k))

    def adtest_Asqr(self,z_list):
        n = len(z_list)
        sum_part = 0
        for idx,j in enumerate(z_list):
            onebase = idx + 1
            left = (2*onebase-1)*math.log(z_list[idx])
            right = (2*n+1-2*onebase)*math.log(1-z_list[idx])
            total = left + right
            sum_part += total
        if n<=5:
            asqr = -1 * n-sum_part/n
            adjasqr = asqr*(1+0.75/n+2.25/(n*n))
            return adjasqr
        else:
            return -1*n-sum_part/n

    def p_val_record(self,Asqr):
        # ref http://www.statisticshowto.com/anderson-darling-test/
        # or more exactly this paper:
        # https://www.jstor.org/stable/1165059?seq=1#page_scan_tab_contents
        if Asqr >= 0.6:
            pval = math.exp(1.2937-5.709*Asqr+0.0186*Asqr*Asqr)
        elif Asqr <0.6 and Asqr > 0.34:
            pval = math.exp(0.9177-4.279*Asqr-1.38*Asqr*Asqr)
        elif Asqr<=0.34 and Asqr > 0.2:
            pval = 1-math.exp(-8.318+42.796*Asqr-59.938*Asqr*Asqr)
        elif Asqr <= 0.2:
            pval = 1-math.exp(-13.436+101.14*Asqr-223.73*Asqr*Asqr)

        if pval>self.pValThres:
            return "fit_good_enough"
        else:
            return "Not_good_enough"