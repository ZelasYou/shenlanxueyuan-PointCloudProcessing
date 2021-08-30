# 文件功能：实现 GMM 算法

import numpy as np
#from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    
    # 屏蔽开始
    # 更新Y
    

    # 更新pi
 
        
    # 更新Mu


    # 更新Var
    def GMM_EM(self):
        '''
        利用EM算法进行优化GMM参数的函数
        :return: 返回数据属于每个分类的概率
        '''
        loglikelyhood = 0
        oldloglikelyhood = 1
        len, dim = np.shape(self.data)
        # gammas[len*n_clusters]后验概率表示第n个样本属于第k个混合高斯的概率，类似K-Means中的rnk，第n个点是否属于第k类
        gammas = np.zeros((len, self.n_clusters))
        # 迭代
        i = 0
        while np.abs(loglikelyhood - oldloglikelyhood) > 1e-4 or i < self.max_iter:
            oldloglikelyhood = loglikelyhood
            # E-STEP 求后验概率gammas， gammas[n]=P(z|x) = P(x|z)P(z) / sum(P(x|z)P(x))
            # 即给定点n，求此点属于各个高斯模型的概率 gammas[n] (1*n_clusters)

            '''
            for n in range(len):
            p_x =  P(x|z)P(z)， p_x_sum = sum( p(x|z)p(x) ) self.weights是先验概率，Gaussian是高斯函数
            p_x = [self.weights[k] * self.Gaussian(self.data[n], self.means[k], self.covars[k])
                    for k in range(self.n_clusters)]
            p_x = np.array(p_x)
            p_x_sum = np.sum(p_x)
            gammas[n] = p_x / p_x_sum
            '''

            for k in range(self.n_clusters):
                prob = multivariate_normal.pdf(self.data, self.means[k], self.covars[k])
                gammas[:, k] = prob * self.weights[k]
                gamma_sum = np.sum(gammas, axis=1)
            for k in range(self.n_clusters):
                gammas[:, k] = gammas[:, k] / gamma_sum
            # M-Step
            for k in range(self.n_clusters):
                # Nk表示样本中有多少概率属于第k个高斯分布
                Nk = np.sum(gammas[:, k])
                # 更新每个高斯分布的权重 pi
                self.weights[k] = 1.0 * Nk / len
                # 更新高斯分布的均值 Mu
                '''self.means[k] = (1.0/Nk)*np.sum([gammas[n][k] * self.data[n] for n in range(len)],axis=0)'''
                self.means[k] = (1.0 / Nk) * np.array([np.sum(gammas[:, k] * self.data[:, d]) for d in range(dim)])
                # 更新高斯分布的协方差矩阵 Var
                xdiffs = self.data - self.means[k]
                '''self.covars[k] = (1.0/Nk) * np.sum([gammas[n][k] * xdiffs[n].reshape((dim,1)).dot(xdiffs[n].reshape((1,dim))) for n in range(len)],axis=0)'''
                # w_xdiffs = w * xdiffs
                w_xdiffs = np.array([gammas[n][k] * xdiffs[n] for n in range(len)])
                self.covars[k] = (1.0 / Nk) * np.dot(w_xdiffs.T, xdiffs)
            loglikelyhood = 0
            '''            
            for n in range(len):
                for k in range(self.n_clusters):
                    loglikelyhood += gammas[n][k]*(np.log(self.weights[k]) + np.log(self.Gaussian(self.data[n],self.means[k],self.covars[k])))
            '''
            for k in range(self.n_clusters):
                loglikelyhood += np.sum(gammas[:, k] * np.log(self.weights[k]) + gammas[:, k] * np.log(
                    multivariate_normal.pdf(self.data, self.means[k], self.covars[k])))
            i += 1



    # 屏蔽结束
    
    def fit(self, data):
        # 作业3
        # 屏蔽开始
        '''
                :param data: 训练数据
                :param n_clusters:高斯分布的个数
                :param weights:每个高斯分布的初始权重，即先验概率P(zk=1)（一个未知的点属于k高斯模型的概率）
                :param means:高斯分布的均值向量
                :param covars:高斯分布的协方差矩阵集合
                '''
        self.data = data
        self.weights = np.random.rand(self.n_clusters)
        self.weights /= np.sum(self.weights)
        dim = np.shape(self.data)[1]
        self.means = []
        # 产生n_clusters个均值
        for i in range(self.n_clusters):
            mean = np.random.rand(dim)
            self.means.append(mean)
        self.covars = []
        # 产生 n_clusters个协方差
        for i in range(self.n_clusters):
            cov = np.eye(dim)
            self.covars.append(cov)
        self.GMM_EM()



        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        len = data.shape[0]
        gammas = np.zeros((len, self.n_clusters))
        for k in range(self.n_clusters):
            prob = multivariate_normal.pdf(self.data, self.means[k], self.covars[k])
            gammas[:, k] = prob * self.weights[k]
            gamma_sum = np.sum(gammas, axis=1)
        for k in range(self.n_clusters):
            gammas[:, k] = gammas[:, k] / gamma_sum
        self.posibility = gammas
        self.prediction = np.array([np.argmax(gammas[i]) for i in range(len)])
        return self.prediction


        # 屏蔽结束

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化

    

