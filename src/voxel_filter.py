# 实现voxel滤波，并加载数据集中的文件进行验证
import open3d as o3d
import os
import numpy as np


# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, method="random"):
    filtered_points = []
    # 获取点云范围
    x_max = np.max(point_cloud[:, 0], axis=0)
    y_max = np.max(point_cloud[:, 1], axis=0)
    z_max = np.max(point_cloud[:, 2], axis=0)
    x_min = np.min(point_cloud[:, 0], axis=0)
    y_min = np.min(point_cloud[:, 1], axis=0)
    z_min = np.min(point_cloud[:, 2], axis=0)

    # 　获取每个维度格子的数量
    nx = ((x_max - x_min) / leaf_size).astype(int)
    ny = ((y_max - y_min) / leaf_size).astype(int)
    nz = ((z_max - z_min) / leaf_size).astype(int)

    # 获取每个点云的index
    i_x = ((point_cloud[:, 0] - x_min) / leaf_size).astype(int)
    i_y = ((point_cloud[:, 1] - x_min) / leaf_size).astype(int)
    i_z = ((point_cloud[:, 2] - x_min) / leaf_size).astype(int)
    index = np.dtype(np.int64)
    index = i_x + i_y * nx + i_z * nx * ny
    # 将索引值与点云数据合并
    point_cloud_idx = np.insert(point_cloud, 0, values=index, axis=1)

    # Sort by the index
    point_cloud_idx = point_cloud_idx[np.lexsort(point_cloud_idx[:, ::-1].T)]
    # print(point_cloud_idx[0:30, :])

    # 　均值或随机voxel降采样
    filtered_points = []
    if method == "centroid":
        k = point_cloud_idx[0, 0]
        n = 0
        for i in range(point_cloud_idx.shape[0]):
            if point_cloud_idx[i, 0] != k:
                filtered_points.append(np.mean(point_cloud_idx[n:i, 1:], axis=0))
                k = point_cloud_idx[i, 0]
                n = i
    elif method == "random":
        k = point_cloud_idx[0, 0]
        n = 0
        for i in range(point_cloud_idx.shape[0]):
            if point_cloud_idx[i, 0] != k:
                random_num = np.random.randint(n, i)  # n ~ i　之间的随机数
                filtered_points.append(point_cloud_idx[random_num, 1:])
                k = point_cloud_idx[i, 0]
                n = i

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points


def main():
    pass


if __name__ == '__main__':
    main()
