# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类
import numpy as np
import open3d as o3d
import struct
import matplotlib.pyplot as plt
from pandas import DataFrame
from pyntcloud import PyntCloud
import math
import random
from sklearn.cluster import DBSCAN


# matplotlib显示点云函数
def Point_Cloud_Show(points):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], cmap='spectral', s=2, linewidths=0, alpha=1, marker=".")
    plt.title('Point Cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    """
    :param path:
    :return: homography matrix of the point cloud, N*3
    """
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    # 初始化数据
    idx_segmented = []
    segmented_cloud = []
    iters = 100  # 最大迭代次数初始化
    sigma = 0.4  # 数据和模型之间可接受的最大差值
    # 模型的参数估计和内点数目,平面方程为   aX + bY + cZ +D= 0
    best_a = 0
    best_b = 0
    best_c = 0
    best_d = 0
    pretotal = 0  # 上一次inline的点数
    # 希望得到正确模型的概率
    P = 0.99
    n = len(data)  # 点的数目
    outline_ratio = 0.6  # 公式中e 一个点位outlier的概率
    for i in range(iters):
        ground_cloud = []
        idx_ground = []
        # step1 选择可以估计出模型的最小数据集，对于平面拟合来说，三个点确定平面
        sample_index = random.sample(range(n), 3)  # 数据集中随机选取3个点
        point1 = data[sample_index[0]]
        point2 = data[sample_index[1]]
        point3 = data[sample_index[2]]
        # step2 求解模型
        # 先求解法向量
        point1_2 = (point1 - point2)  # 向量 point1 -> point2
        point1_3 = (point1 - point3)  # 向量 point1 -> point3
        N = np.cross(point1_3, point1_2)  # 向量叉乘求解 平面法向量
        # slove model 求解模型的a,b,c,d
        a = N[0]
        b = N[1]
        c = N[2]
        d = -N.dot(point1)
        # step3 将所有数据带入模型，计算出“内点”的数目；(累加在一定误差范围内的适合当前迭代推出模型的数据)
        total_inlier = 0
        pointn_1 = (data - point1)  # sample（三点）外的点 与 sample内的三点其中一点 所构成的向量
        distance = abs(pointn_1.dot(N)) / np.linalg.norm(N)  # 求距离
        # 使用距离判断inline
        idx_ground = (distance <= sigma)
        total_inlier = np.sum(idx_ground == True)  # 统计inline得点数
        # 判断当前的模型是否比之前估算的模型好
        if total_inlier > pretotal:  # log(1 - p)
            iters = math.log(1 - P) / math.log(1 - pow(total_inlier / n, 3))
            # 当前正确概率，动态调整
            # 根据公式计算新的迭代次数
            pretotal = total_inlier  # 重新幅值
            # 获取最好得 abcd 模型参数
            best_a = a
            best_b = b
            best_c = c
            best_d = d

        # 判断是否当前模型已经符合条件
        if total_inlier > n * (1 - outline_ratio):
            break
    print("iters = %f" % iters)
    # 提取分割后得点
    idx_segmented = np.logical_not(idx_ground)
    ground_cloud = data[idx_ground]
    segmented_cloud = data[idx_segmented]
    return ground_cloud, segmented_cloud

    # 屏蔽结束


# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 使用sklearn dbscan库
    # eps 两个样本的最大距离，即扫描半径；
    # min_samples 作为核心点的话邻域(即以其为圆心，eps为半径的圆，含圆上的点)中的最小样本数(包括点本身);
    # n_jobs ：-1使用CPU所有核心
    cluster_index = DBSCAN(eps=0.25, min_samples=5, n_jobs=-1).fit_predict(data)
    print(cluster_index)
    return cluster_index
    # 屏蔽结束


# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(segmented_ground, segmented_cloud, cluster_index):
    def colormap(c, num_clusters):
        # outlier:
        if c == -1:
            color = [1] * 3
        # surrouding object:
        else:
            color = [0] * 3
            color[c % 3] = c / num_clusters

        return color

    # ground element:
    pcd_ground = o3d.geometry.PointCloud()
    pcd_ground.points = o3d.utility.Vector3dVector(segmented_ground)
    pcd_ground.colors = o3d.utility.Vector3dVector(
        [
            [0, 0, 255] for i in range(segmented_ground.shape[0])
        ]
    )

    # surrounding object elements:
    pcd_objects = o3d.geometry.PointCloud()
    pcd_objects.points = o3d.utility.Vector3dVector(segmented_cloud)
    num_clusters = max(cluster_index) + 1
    pcd_objects.colors = o3d.utility.Vector3dVector(
        [
            colormap(c, num_clusters) for c in cluster_index
        ]
    )

    # visualize:
    o3d.visualization.draw_geometries([pcd_ground, pcd_objects])


def main():
    filename = 'E:\\PointCloudSourceFile\\bin\\KITTI_object\\007476.bin'  # 数据集路径
    print('clustering pointcloud file:', filename)

    origin_points = read_velodyne_bin(filename)  # 读取数据点
    origin_points_df = DataFrame(origin_points, columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_pynt = PyntCloud(origin_points_df)  # 将points的数据 存到结构体中
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化

    # 地面分割
    ground_points, segmented_points = ground_segmentation(data=origin_points)
    # ground_points是地面点云，segmented_points是除地面外的其他需要聚类分割的点云
    ground_points_df = DataFrame(ground_points, columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_pynt_ground = PyntCloud(ground_points_df)  # 将points的数据 存到结构体中
    point_cloud_o3d_ground = point_cloud_pynt_ground.to_instance("open3d", mesh=False)  # 实例化
    point_cloud_o3d_ground.paint_uniform_color([0, 0, 255])  # RGB数值为蓝色，代表地面

    # 显示待分割聚类的点云
    segmented_points_df = DataFrame(segmented_points, columns=['x', 'y', 'z'])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_pynt_segmented = PyntCloud(segmented_points_df)  # 将points的数据 存到结构体中
    point_cloud_o3d_segmented = point_cloud_pynt_segmented.to_instance("open3d", mesh=False)  # 实例化
    point_cloud_o3d_segmented.paint_uniform_color([255, 0, 0])  # RGB为红色，代表车辆、行人等其他物体
    # 显示地面与其他物体分割效果
    o3d.visualization.draw_geometries([point_cloud_o3d_ground,point_cloud_o3d_segmented])

    # 显示聚类结果
    cluster_index = clustering(segmented_points)
    plot_clusters(ground_points, segmented_points, cluster_index)


if __name__ == '__main__':
    main()
