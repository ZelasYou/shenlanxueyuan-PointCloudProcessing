# kdtree的具体实现，包括构建和查找

import random
import math
import numpy as np

from result_set import KNNResultSet, RadiusNNResultSet


# Node类，Node是tree的基本组成元素
class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis  # 划分的维度
        self.value = value  # 节点的值
        self.left = left  # 左节点
        self.right = right  # 右节点
        self.point_indices = point_indices  # 点集中的信号

    # 检查是否是叶子节点（函数）
    def is_leaf(self):
        if self.value is None:  # 如果返回值为空，那么这个是叶子
            return True
        else:
            return False

    # 打印函数
    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '  # 分割值：叶
        else:
            output += 'split value: %.2f, ' % self.value  # 保留后两位小数
        output += 'point_indices: '  # 点指数
        output += str(self.point_indices.tolist())  # str函数，转换为人阅读比较舒服的形式。 tolist:将数组转化为列表
        return output


# 功能：构建树之前需要对value进行排序，同时对一个的key的顺序也要跟着改变
# 输入：
#     key：键
#     value:值
# 输出：
#     key_sorted：排序后的键
#     value_sorted：排序后的值
def sort_key_by_vale(key, value):
    assert key.shape == value.shape  # arrest：条件为value的维度等于key的维度的时候继续执行
    assert len(key.shape) == 1  # 条件为键的维度为1的时候继续执行
    sorted_idx = np.argsort(value)  # 返回数组值从小到大的索引值，例如：[3,1,2]返回[1,2,0]
    key_sorted = key[sorted_idx]  # 返回排序后的key
    value_sorted = value[sorted_idx]  # 返回排序后的value
    return key_sorted, value_sorted


# 改变分割维度：轮换确定分割维度：0-->1,1-->2,2-->0
def axis_round_robin(axis, dim):  # 输入维度、
    if axis == dim - 1:
        return 0
    else:
        return axis + 1


# 功能：通过递归的方式构建树
# 输入：
#     root: 树的根节点
#     db: 点云数据
#     point_indices：排序后的键
#     axis: scalar  标量维度
#     leaf_size: scalar 标量叶子规格
# 输出：
#     root: 即构建完成的树


def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)  # 如果root不存在那么先建立root

    # determine whether to split into left and right 决定是否分成左右两部分
    # 如果节点数大于叶子节点数，才进一步进行分割，否则就不进行进一步分割，当前节点就作为叶子节点
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        # 对当前传入的数据节点在划分维度上进行排序，选出当前维度的中间值数据点
        # 划分点的value就等于中间值数据点的均值，注意此处划分的中间平面不穿过数据点
        point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])  # 点的索引按照value的大小进行排序
        # 作业1
        # 屏蔽开始

        # 求出当前维度下中间值的点的索引位置
        middle_left_idx = math.ceil(point_indices_sorted.shape[0] / 2) - 1
        # 中间点在原来点集合中的索引
        middle_left_point_idx = point_indices_sorted[middle_left_idx]
        # 中间点的value值
        middle_left_point_value = db[middle_left_point_idx, axis]
        # 中间点右面的点也一样
        middle_right_idx = middle_left_idx + 1
        middle_right_point_idx = point_indices_sorted[middle_right_idx]
        middle_right_point_value = db[middle_right_point_idx, axis]

        root.value = (middle_left_point_value + middle_right_point_value) * 0.5  # 取中间两个数据点value的平均值，不穿过数据点
        # === get the split position === 以下为迭代寻找
        root.left = kdtree_recursive_build(root.left, db, point_indices_sorted[0:middle_right_idx],
                                           axis_round_robin(axis, dim=db.shape[1]), leaf_size)
        root.right = kdtree_recursive_build(root.right, db, point_indices_sorted[middle_right_idx:],
                                            axis_round_robin(axis, dim=db.shape[1]), leaf_size)
        # 屏蔽结束
    return root


# 功能：翻转一个kd树
# 输入：
#     root：kd树
#     depth: 当前深度
#     max_depth：最大深度
def traverse_kdtree(root: Node, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():
        print(root)
    else:
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1


# 功能：构建kd树（利用kdtree_recursive_build功能函数实现的对外接口）
# 输入：
#     db_np：原始数据
#     leaf_size：scale 标量
# 输出：
#     root：构建完成的kd树、
def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root, db_np, np.arange(N), axis=0, leaf_size=leaf_size)
    return root


# 功能：通过kd树实现knn搜索，即找出最近的k个近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set：搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False  # 如果没有建立KD树，那么立刻返回失败

    if root.is_leaf():
        # compare the contents of a leaf 比较树叶的内容
        leaf_points = db[root.point_indices, :]  # root.point_indeces是一个列表，存储的是点在元数据中的索引
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # 作业2
    # 提示：仍通过递归的方式实现搜索
    # 屏蔽开始
    # 距离小于root.value，从左边开始找
    if query[root.axis] <= root.value:
        kdtree_knn_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():  # 如果左边没有找够，就继续从右边找
            kdtree_knn_search(root.right, db, result_set, query)
    else:
        # 从右边开始找
        kdtree_knn_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_knn_search(root.left, db, result_set, query)

    # 屏蔽结束

    return False


# 功能：通过kd树实现radius搜索，即找出距离radius以内的近邻
# 输入：
#     root: kd树
#     db: 原始数据
#     result_set:搜索结果
#     query：索引信息
# 输出：
#     搜索失败则返回False
def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    # 作业3
    # 提示：通过递归的方式实现搜索
    # 屏蔽开始
    if query[root.axis] <= root.value:
        kdtree_radius_search(root.left, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.right, db, result_set, query)
    else:
        kdtree_radius_search(root.right, db, result_set, query)
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist():
            kdtree_radius_search(root.left, db, result_set, query)

    # 屏蔽结束

    return False


def main():
    # configuration
    db_size = 64
    dim = 3
    leaf_size = 4
    k = 1

    db_np = np.random.rand(db_size, dim)

    root = kdtree_construction(db_np, leaf_size=leaf_size)

    depth = [0]
    max_depth = [0]
    traverse_kdtree(root, depth, max_depth)
    print("tree max depth: %d" % max_depth[0])

    # query = np.asarray([0, 0, 0])
    # result_set = KNNResultSet(capacity=k)
    # knn_search(root, db_np, result_set, query)
    #
    # print(result_set)
    #
    # diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    # nn_idx = np.argsort(diff)
    # nn_dist = diff[nn_idx]
    # print(nn_idx[0:k])
    # print(nn_dist[0:k])
    #
    #
    # print("Radius search:")
    # query = np.asarray([0, 0, 0])
    # result_set = RadiusNNResultSet(radius = 0.5)
    # radius_search(root, db_np, result_set, query)
    # print(result_set)


if __name__ == '__main__':
    main()
