import ground_seg
import voxel_filter as vfd
import clustering
import open3d as o3d
import os
import numpy as np
import struct
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def read_velodyne_bin(path):
    '''从kitti的.bin格式点云文件中读取点云
    Args:
        path: 文件路径

    Returns:
        homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


def plot_pcd(points, index=None):
    pcd = o3d.geometry.PointCloud()
    if index is not None:
        pcd.points = o3d.utility.Vector3dVector(points[index])
    else:
        pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, y_pred):
    fig = plt.figure(figsize=(50,30))
    ax = Axes3D(fig)
    colors = np.array(
        list(
            islice(
                cycle(
                    [
                        "#377eb8",
                        "#ff7f00",
                        "#4daf4a",
                        "#f781bf",
                        "#a65628",
                        "#984ea3",
                        "#999999",
                        "#e41a1c",
                        "#dede00",
                    ]
                ),
                int(max(y_pred) + 1),
            )
        )
    )
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=10, color=colors[y_pred])
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    #ax.view_init(20, -5) # elev, azim
    plt.show()


if __name__ == '__main__':
    root_dir = '/home/yst/Documents/groundSegAndClustering/velodyne_data'
    cat = os.listdir(root_dir)
    cat = cat[:]
    iteration_num = len(cat)
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        data = read_velodyne_bin(filename)
        print('-------------step1 : origin data ----------------')
        # 显示原始点云
        plot_pcd(data)

        print('----------------step2 : ground segmentation -------------------')
        # 显示地面点和非地面点
        # RANSAC
        ransac = ground_seg.RANSAC(eps=0.25, max_iter=300, inliers_ratio=0.6, n=3)
        ground_points_idx, non_ground_points_idx = ransac.get_model(data)
        plot_pcd(data, ground_points_idx)
        plot_pcd(data, non_ground_points_idx)

        print('-------------step3 : voxel downsampling ------------')
        # 降采样
        # voxel grid filter
        filtered_points = vfd.voxel_filter(data[non_ground_points_idx], leaf_size=0.2)
        plot_pcd(filtered_points)

        print('------------step4 : clustering -----------')
        # 聚类非地面点
        # DBSCAN
        non_ground_points = data[non_ground_points_idx]

        print('------------please wait a moment -----------')
        dbscan = clustering.OwnDBSCAN(radius=1.0, min_samples=8)
        dbscan.fit(non_ground_points)
        plot_clusters(non_ground_points, dbscan.labels_.astype(int))

        dbscan.fit(filtered_points)
        plot_clusters(filtered_points, dbscan.labels_.astype(int))
