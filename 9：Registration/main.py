import os
import collections

import open3d as o3d
# IO utils:
import utils.io as io
import utils.visualize as visualize

# 导入iss feature detection 模块
from iss import detect

from RANSAC import ransac_match
from ICP import icp_exact_match

# RANSAC configuration:
RANSACParams = collections.namedtuple(
    'RANSACParams',
    [
        'max_workers',
        'num_samples',
        'max_correspondence_distance', 'max_iteration', 'max_validation', 'max_refinement'
    ]
)
# fast pruning algorithm configuration:
CheckerParams = collections.namedtuple(
    'CheckerParams',
    ['max_correspondence_distance', 'max_edge_length_ratio', 'normal_angle_threshold']
)

if __name__ == '__main__':
    radius = 0.5
    # step1 读取数据
    input_dir = "registration_dataset"
    registration_results = io.read_registration_results(os.path.join(input_dir,
                                                                     'E:\\PointCloudSourceFile\\bin\\registration_dataset\\reg_result.txt'))  # 读取 reg_result.txt 结果
    ##init output
    df_output = io.init_output()
    ##批量读取
    # for i,r  in (
    #         list(registration_results.iterrows())
    # ):
    #     idx_target = int(r['idx1'])
    #     idx_source = int(r['idx2'])
    idx_target = 643
    idx_source = 456
    ##load point cloud
    pcd_source = io.read_point_cloud_bin(
        os.path.join(input_dir, 'E:\\PointCloudSourceFile\\bin\\registration_dataset\\point_clouds',
                     f'{idx_source}.bin')
    )
    pcd_target = io.read_point_cloud_bin(
        os.path.join(input_dir, 'E:\\PointCloudSourceFile\\bin\\registration_dataset\\point_clouds',
                     f'{idx_target}.bin')
    )
    # step2 对点云进行降采样downsample
    pcd_source, idx_inliers = pcd_source.remove_radius_outlier(nb_points=4, radius=radius)
    pcd_target, idx_inliers = pcd_target.remove_radius_outlier(nb_points=4, radius=radius)
    ## 构建 kd-tree
    search_tree_source = o3d.geometry.KDTreeFlann(pcd_source)
    search_tree_target = o3d.geometry.KDTreeFlann(pcd_target)
    # step3 iss 特征点提取
    keypoints_source = detect(pcd_source, search_tree_source, radius)
    keypoints_target = detect(pcd_target, search_tree_target, radius)
    # step4 fpfh特征点描述 feature description
    pcd_source_keypoints = pcd_source.select_by_index(keypoints_source['id'].values)
    ##fpfh 进行 特征点描述
    fpfh_source_keypoints = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_source_keypoints,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 5, max_nn=100)
    ).data
    pcd_target_keypoints = pcd_target.select_by_index(keypoints_target['id'].values)
    fpfh_target_keypoints = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_target_keypoints,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 5, max_nn=100)
    ).data

    # generate matches:
    distance_threshold_init = 1.5 * radius
    distance_threshold_final = 1.0 * radius
    # step5 RANSAC Registration,初配准、得到初始旋转、平移矩阵
    init_result = ransac_match(
        idx_target, idx_source,
        pcd_source_keypoints, pcd_target_keypoints,
        fpfh_source_keypoints, fpfh_target_keypoints,
        ransac_params=RANSACParams(
            max_workers=5,
            num_samples=4,
            max_correspondence_distance=distance_threshold_init,
            max_iteration=200000,
            max_validation=500,
            max_refinement=30
        ),
        checker_params=CheckerParams(
            max_correspondence_distance=distance_threshold_init,
            max_edge_length_ratio=0.9,
            normal_angle_threshold=None
        )
    )
    # step6 ICP for refined estimation,优化配准 ICP对初始T进行迭代优化:
    final_result = icp_exact_match(
        pcd_source, pcd_target, search_tree_target,
        init_result.transformation,
        distance_threshold_final, 60
    )
    # visualize:
    visualize.show_registration_result(
        pcd_source_keypoints, pcd_target_keypoints, init_result.correspondence_set,
        pcd_source, pcd_target, final_result.transformation
    )
    #
    # # add result:
    # io.add_to_output(df_output, idx_target, idx_source, final_result.transformation)
    #
    # # write output:
    # io.write_output(
    #     os.path.join(input_dir, 'reg_result_yaogefad.txt'),
    #     df_output
    # )
