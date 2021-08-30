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

def shall_terminate(result_curr, result_prev):
    # relative fitness improvement:
    relative_fitness_gain = result_curr.fitness / result_prev.fitness - 1

    return relative_fitness_gain < 0.01

def icp_exact_match(
        pcd_source, pcd_target, search_tree_target,
        T,
        max_correspondence_distance, max_iteration
):
    # num. points in the source:
    N = len(pcd_source.points)

    # evaluate relative change for early stopping:
    result_prev = result_curr = o3d.pipelines.registration.evaluate_registration(
        pcd_source, pcd_target, max_correspondence_distance, T
    )

    for _ in range(max_iteration):
        # TODO: transform is actually an in-place operation. deep copy first otherwise the result will be WRONG
        pcd_source_current = copy.deepcopy(pcd_source)
        # apply transform:
        pcd_source_current = pcd_source_current.transform(T)

        # find correspondence:
        matches = []
        for n in range(N):
            query = np.asarray(pcd_source_current.points)[n]
            _, idx_nn_target, dis_nn_target = search_tree_target.search_knn_vector_3d(query, 1)

            if dis_nn_target[0] <= max_correspondence_distance:
                matches.append(
                    [n, idx_nn_target[0]]
                )
        matches = np.asarray(matches)

        #icp
        if len(matches) >= 4:
            # sovle ICP:
            P = np.asarray(pcd_source.points)[matches[:, 0]]
            Q = np.asarray(pcd_target.points)[matches[:, 1]]
            T = solve_procrustes_transf(P, Q)

            # evaluate:
            result_curr = o3d.pipelines.registration.evaluate_registration(
                pcd_source, pcd_target, max_correspondence_distance, T
            )

            # if no significant improvement:提前中止
            if shall_terminate(result_curr, result_prev):
                print('[RANSAC ICP]: Early stopping.')
                break

    return result_curr

