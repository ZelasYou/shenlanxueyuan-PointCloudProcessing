# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import os
import struct

import octree
import kdtree
from result_set import KNNResultSet, RadiusNNResultSet


def read_velodyne_bin(path):  # 读取.bin的函数
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []  # 定义一个空列表
    with open(path, 'rb') as f: # “rb：以二进制形式只读非文本文件”，指针放在文件的开头
        content = f.read()  # 读取文件
        pc_iter = struct.iter_unpack('ffff', content)   # 在文件当中每次读取4位浮点数，直到整个文件全部读取完成
        for idx, point in enumerate(pc_iter):  # 遍历整个列表，将返回idx（元素位置）和point（元素）
            pc_list.append([point[0], point[1], point[2]])  # 在列表末尾添加新的对象
    return np.asarray(pc_list, dtype=np.float32).T  # 以整形32位将数据的转置存储


def main():
    # configuration
    leaf_size = 32  # 每个小立方体支持的最大的点云数量
    min_extent = 0.0001  # 最小划分区间为0.0001
    k = 8  # 8个最邻近点
    radius = 1

    root_dir = 'E:\\PointCloudSourceFile\\bin'  # 数据集路径
    cat = os.listdir(root_dir)  # 返回所有数据集文件和文件夹列表
    iteration_num = len(cat)  # 返回文件及文件夹数量

    print("octree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)  # 遍历所有.bin文件

        begin_t = time.time()  # 开始计时
        root = octree.octree_construction(db_np, leaf_size, min_extent)  # 输入 点云数据、单位体积最大点云存储数量、最小划分区间，输出root（）
        # root（root、点云数据、点云位置中心值、最大距离半径、点云计数器、单位体积最大点云存储数量、最小划分区间）
        construction_time_sum += time.time() - begin_t  # 计算时间差

        query = db_np[0, :]   # 读取点云X列

        begin_t = time.time()  # 再次计时
        result_set = KNNResultSet(capacity=k)  # 创建一个类，默认capacity=k
        octree.octree_knn_search(root, db_np, result_set, query)
        knn_time_sum += time.time() - begin_t  # 计算所花时间

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        octree.octree_radius_search_fast(root, db_np, result_set, query)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     brute_time_sum * 1000 / iteration_num))

    print("kdtree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        db_np = read_velodyne_bin(filename)

        begin_t = time.time()
        root = kdtree.kdtree_construction(db_np, leaf_size)
        construction_time_sum += time.time() - begin_t

        query = db_np[0, :]

        begin_t = time.time()
        result_set = KNNResultSet(capacity=k)
        kdtree.kdtree_knn_search(root, db_np, result_set, query)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        result_set = RadiusNNResultSet(radius=radius)
        kdtree.kdtree_radius_search(root, db_np, result_set, query)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t
    print("Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000 / iteration_num,
                                                                     knn_time_sum * 1000 / iteration_num,
                                                                     radius_time_sum * 1000 / iteration_num,
                                                                     brute_time_sum * 1000 / iteration_num))


if __name__ == '__main__':
    main()
