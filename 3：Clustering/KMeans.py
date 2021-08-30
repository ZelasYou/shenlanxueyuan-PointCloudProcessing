# 文件功能： 实现 K-Means 算法

import numpy as np
import random


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.centers = None
        self.fitted = False

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        # 随机选取数据中的中心点
        centers = data[random.sample(range(data.shape[0]), self.k_)]
        # 从原始数据中随机抽取k个作为一个列表
        old_centers = np.copy(centers)  # 得到初始中心点，完全为随机数据点
        # 以下代码是生成k个空列表
        labels = [[] for i in range(self.k_)]
        for iters in range(self.max_iter_):  # 循环最大次数
            for idx, point in enumerate(data):
                # 以norm函数计算距离
                diff = np.linalg.norm(old_centers - point, axis=1)
                # np.argmin(diff) 表示最小值在数组中的位置，距离最小的索引就是类
                diff2 = (np.argmin(diff))
                labels[diff2].append(idx)

            for i in range(self.k_):
                points = data[labels[i], :]  # 所有在第k类中的所有点
                centers[i] = points.mean(axis=0)  # 该类所有点的均值作为聚类中心
            if np.sum(np.abs(
                    centers - old_centers)) < self.tolerance_ * self.k_:
                # 若此时迭代距离已经几乎不变，则跳出循环，迭代结束
                break
            old_centers = np.copy(centers)  # 每次迭代结果更新
        self.centers = centers
        self.fitted = True
        # 屏蔽结束

    def predict(self, p_datas):  # 预测每个点属于的类别
        result = []
        # 作业2
        # 屏蔽开始
        if not self.fitted:
            print('尚未聚类成功')
            return result
        for point in p_datas:
            diff = np.linalg.norm(self.centers - point, axis=1)
            result.append(np.argmin(diff))
        # 屏蔽结束
        return result


if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)
