# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d  # 添加open3d库并设置快捷名称o3d
import numpy as np  # 添加numpy库并设置快捷名称为np
from scipy.spatial import KDTree  # 添加KD树库


# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量


def PCA(data, correlation=False, sort=True):
    # 作业1
    # 设m行n维矩阵，m=点云数，n=3
    # 组成n*m的矩阵
    # 求出X行的每一个平均值
    # 将每一行减去这一行的平均值
    # 求出协方差矩阵
    # 求出协方差矩阵的特征值和特征向量
    # 将特征向量按对应特征值大小从上到下按行排列成矩阵

    data_T = data.T  # 数组转置
    s = np.array(data_T)  # 获取数组的行列数
    n = s.shape[0]  # 获取行数（x,y,z）
    m = s.shape[1]  # 获取列数（点云数）
    mean = [0] * 3  # 定义一个平均值空数组

    for i in range(n):  # 进行行数循环
        mean[i] = np.mean(data_T[i, :])  # 求出每行的平均值
        for j in range(m):  # 进行列数循环
            data_T[i, j] -= mean[i]  # 减去平均值
    dataTT = data_T.T  # 转置修改后的数组
    c = 1 / m * np.matmul(data_T, dataTT)  # 协方差c
    eigenvalues, eigenvectors = np.linalg.eig(c)  # 求出矩阵的特征值和特征向量
    # 判断是否排序
    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors  # 返回特征值和特征向量（注意是3*3的）


def main():
    filename = "E:\\PointCloudSourceFile\\txt\\modeled40_normal_resampled\\sofa\\sofa_0011.txt"  # 指定点云路径
    points = np.loadtxt(filename, delimiter=',')  # 读取txt文件，分隔符为逗号
    xyz = points[:, :3]  # 提取x，y，z
    w, v = PCA(xyz)  # PCA点云

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame()  # 在原点创建坐标轴网络
    axis.rotate(v, center=(0, 0, 0))  # 按特征向量进行坐标轴旋转
    #p_xyz = o3d.utility.Vector3dVector(xyz)  # 定义点云坐标位置
    #pc_v = o3d.geometry.PointCloud(p_xyz)  # 定义点云
    #o3d.visualization.draw_geometries([pc_v])  # 单独显示点云的界面
    #o3d.visualization.draw_geometries([pc_v, axis], point_show_normal=True)  # 显示点云和三个特征向量组成的坐标轴

    # 循环计算每个点的法向量
    leafsize = 32  # 切换为暴力搜索的最小数量
    KDTree_radius = 0.1  # 设置邻域半径
    tree = KDTree(xyz, leafsize=leafsize)  # 构建KDTree
    radius_neighbor_idx = tree.query_ball_point(xyz, KDTree_radius)  # 得到每个点的邻近索引
    normals = []  # 定义一个空list
    # -------------寻找法线---------------
    # 首先寻找邻域内的点
    for i in range(len(radius_neighbor_idx)):
        neighbor_idx = radius_neighbor_idx[i]  # 得到第i个点的邻近点索引，邻近点包括自己
        neighbor_data = xyz[neighbor_idx]  # 得到邻近点，在求邻近法线时没必要归一化，在PCA函数中归一化就行了
        eigenvalues, eigenvectors = PCA(neighbor_data)  # 对邻近点做PCA，得到特征值和特征向量
        normals.append(eigenvectors[:, 2])  # 最小特征值对应的方向就是法线方向
    # ------------法线查找结束---------------
    normals = np.array(normals, dtype=np.float64)  # 把法线放在了normals中
    # o3d.geometry.PointCloud，返回了PointCloud类型
    pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(xyz))
    # 向PointCloud对象中添加法线
    pc_view.normals = o3d.utility.Vector3dVector(normals)
    # 可视化
    o3d.visualization.draw_geometries([pc_view, axis], point_show_normal=True)
    # o3d.io.write_image("bathtub_0008.jpg", img)


if __name__ == '__main__':
    main()  # 类似于运行主程序的意思
