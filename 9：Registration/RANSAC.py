import os
import argparse
import progressbar
import collections
import numpy as np
import open3d as o3d
import copy
import concurrent.futures
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
#IO utils:
import utils.io as io
import utils.visualize as visualize
import matplotlib.pyplot as plt

from ICP import  icp_exact_match

def solve_procrustes_transf(P,Q):    # 求解平移旋转矩阵 solve_procrustes_transformation
    up = P.mean(axis=0)
    uq = Q.mean(axis=0)

    # move to center:
    P_centered = P - up
    Q_centered = Q - uq

    U, s, V = np.linalg.svd(np.dot(Q_centered.T, P_centered), full_matrices=True, compute_uv=True)
    R = np.dot(U, V)
    t = uq - np.dot(R, up)

    # format as transform:
    T = np.zeros((4, 4))
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    T[3, 3] = 1.0

    return T

def get_init_matches(feature_source, feature_target):
    ##对 feature target fpfh 建立 kd—tree
    fpfh_search_tree = o3d.geometry.KDTreeFlann(feature_target)
    ##建立 pairs
    _,N = feature_source.shape
    matches = []
    for i in range(N):
        query = feature_source[:,i]
        _, idx_nn_target, _ = fpfh_search_tree.search_knn_vector_xd(query, 1)   #source -> target
        matches.append([i,idx_nn_target[0]])    #通过knn 寻找唯一 的 nearest points 一一配对 构建pair

    matches = np.asarray(matches)
    return  matches

def iter_match(
    idx_target,idx_source,
    pcd_source, pcd_target,
    proposal,
    checker_params
):
    idx_source, idx_target = proposal[:, 0], proposal[:, 1]
    #法向量校准
    if not checker_params.normal_angle_threshold is None:
        # get corresponding normals:
        normals_source = np.asarray(pcd_source.normals)[idx_source]
        normals_target = np.asarray(pcd_target.normals)[idx_target]

        # a. normal direction check:
        normal_cos_distances = (normals_source*normals_target).sum(axis = 1)
        is_valid_normal_match = np.all(normal_cos_distances >= np.cos(checker_params.normal_angle_threshold))

        if not is_valid_normal_match:
            return None

    # get corresponding points:
    points_source = np.asarray(pcd_source.points)[idx_source]
    points_target = np.asarray(pcd_target.points)[idx_target]

    # b. edge length ratio check:
    #构建距离矩阵，使用 Mutual nearest descriptor matching
    pdist_source = pdist(points_source)
    pdist_target = pdist(points_target)
    is_valid_edge_length = np.all(
        np.logical_and(
            pdist_source > checker_params.max_edge_length_ratio * pdist_target,
            pdist_target > checker_params.max_edge_length_ratio * pdist_source
        )
    )

    if not is_valid_edge_length:
        return None

    # c. fast correspondence distance check:s
    T = solve_procrustes_transf(points_source, points_target)    #通过 svd 初步求解 旋转、平移矩阵
    R, t = T[0:3, 0:3], T[0:3, 3]
    #deviation：偏差  区分 inline outline 通过 距离判断
    deviation = np.linalg.norm(
        points_target - np.dot(points_source, R.T) - t,
        axis = 1
    )
    #判断数目
    is_valid_correspondence_distance = np.all(deviation <= checker_params.max_correspondence_distance)

    return T if is_valid_correspondence_distance else None

def ransac_match(
        idx_target,idx_source,
        pcd_source, pcd_target,
        feature_source, feature_target,
        ransac_params, checker_params
):
    #step5.1 Establish correspondences(point pairs) 建立 pairs
    matches = get_init_matches(feature_source, feature_target)   #通过 fpfh 建立的feature squre map 建立最初的 pairs

    ##build search tree on the target:
    search_tree_target = o3d.geometry.KDTreeFlann(pcd_target)

    N, _ = matches.shape
    idx_matches = np.arange(N)   #对每队 pair 打标签

    T = None                     #translation martix
    #step5.2 select 4 pairs at each iteration,选择4对corresponding 进行模型拟合
    proposal_generator = (
        matches[np.random.choice(idx_matches, ransac_params.num_samples, replace=False)] for _ in iter(int, 1)
    )
    ##step5.3 iter 迭代， iter_match() ,选择出 vaild T
    validator = lambda proposal: iter_match(idx_target,idx_source,pcd_source, pcd_target, proposal, checker_params)

    with concurrent.futures.ThreadPoolExecutor(max_workers=ransac_params.max_workers) as executor:
        for T in map(
                validator,
                proposal_generator        #map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
        ):
            print(T)
            if not (T is None):
                break

    #set baseline
    print('[RANSAC ICP]: Get first valid proposal. Start registration...')
    best_result = icp_exact_match(
        pcd_source, pcd_target, search_tree_target,
        T,
        ransac_params.max_correspondence_distance,
        ransac_params.max_refinement
    )

    # RANSAC:
    num_validation = 0
    for i in range(ransac_params.max_iteration):
        # get proposal:
        T = validator(next(proposal_generator))

        # check validity:
        if (not (T is None)) and (num_validation < ransac_params.max_validation):
            num_validation += 1

            # refine estimation on all keypoints:
            result = icp_exact_match(
                pcd_source, pcd_target, search_tree_target,
                T,
                ransac_params.max_correspondence_distance,
                ransac_params.max_refinement
            )

            # update best result:
            best_result = best_result if best_result.fitness > result.fitness else result

            if num_validation == ransac_params.max_validation:
                break

    return best_result
