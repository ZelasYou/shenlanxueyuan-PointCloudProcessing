# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d
import numpy as np


# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸



def voxel_filter(point_cloud, leaf_size):
    filtered_points = []

    minPC = np.min(point_cloud, axis=0)  # 找到点云的各列最小值
    maxPC = np.max(point_cloud, axis=0)  # 找到点云的各列最大值
    Dx = (maxPC[0] - minPC[0]) // leaf_size + 1  # 横向盒子的数量
    Dy = (maxPC[1] - minPC[1]) // leaf_size + 1  # 纵向盒子的数量
    Dz = (maxPC[2] - minPC[2]) // leaf_size + 1  # 立向盒子的数量
    print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))  # 打印一下盒子数量

    # 计算每个点的voxel索引
    h = list()  # h 为保存索引的列表
    for i in range(len(point_cloud)):
        hx = (point_cloud[i][0] - minPC[0]) // leaf_size
        hy = (point_cloud[i][1] - minPC[1]) // leaf_size
        hz = (point_cloud[i][2] - minPC[2]) // leaf_size
        h.append(hx + hy * Dx + hz * Dx * Dy)
    h = np.array(h)

    # 筛选点
    h_indice = np.argsort(h)  # 返回h里面的元素按从小到大排序的索引
    h_sorted = h[h_indice]
    begin = 0
    for i in range(len(h_sorted) - 1):  # 0~9999
        if h_sorted[i] == h_sorted[i + 1]:
            continue
        else:
            point_idx = h_indice[begin: i + 1]
            filtered_points.append(np.mean(point_cloud[point_idx], axis=0))
            begin = i + 1

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


def main():
    # 加载自己的点云文件
    filename = "E:\\PointCloudSourceFile\\txt\\modeled40_normal_resampled\\car\\car_0008.txt"  # 指定点云路径
    points = np.loadtxt(filename, delimiter=',')  # 读取txt文件，分隔符为逗号
    xyz = points[:, :3]  # 提取x，y，z

    p_xyz = o3d.utility.Vector3dVector(xyz)  # 定义点云坐标位置
    pc_v = o3d.geometry.PointCloud(p_xyz)  # 定义点云
    o3d.visualization.draw_geometries([pc_v])  # 单独显示点云的界面

    filtered_cloud = voxel_filter(xyz, 0.05)  # 调用voxel滤波函数，实现滤波

    v_xyz = o3d.utility.Vector3dVector(filtered_cloud)  # 定义点云坐标位置
    pv_v = o3d.geometry.PointCloud(v_xyz)  # 定义点云
    o3d.visualization.draw_geometries([pv_v])  # 显示滤波后的点云


if __name__ == '__main__':
    main()
