import math
import numpy as np
import time
from result_set import KNNResultSet, RadiusNNResultSet
import os
import struct


def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)  # 对二进制文件循环解包并返回索引
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)


class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis_ = axis
        self.value_ = value
        self.left_ = left
        self.right_ = right
        self.point_indices_ = point_indices

    def is_leaf(self):
        if self.value_ is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis_
        if self.value_ is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value_
        output += 'point_indices: '
        output += str(self.point_indices_.tolist())
        return output


class KDTree_opt:
    def __init__(self, data, leaf_size=5, dimension=3, sample_size=100):
        assert leaf_size >= 1
        self.data_ = data
        self.leaf_size_ = leaf_size
        self.dimension_ = dimension
        self.sample_size_ = sample_size
        self.root_ = self.construct_tree()

    def construct_tree(self):
        root = None
        root = self.recursive_build_tree(root, self.data_,
                                         np.arange(self.data_.shape[0]),
                                         axis=0, leaf_size=self.leaf_size_)
        return root

    def recursive_build_tree(self, root, data, point_indices, axis, leaf_size):
        if root is None:
            root = Node(axis, None, None, None, point_indices)

        if len(point_indices) > leaf_size:
            point_indices_sorted, point_value_sorted, median_point_index, median_point_val = \
                self.get_partition(point_indices, data[point_indices, axis])

            root.value_ = median_point_val
            axis = self.next_axis(data[point_indices_sorted])  # 按方差最大的方向进行划分，加快搜索速度，但会降低构建速度
            root.left_ = self.recursive_build_tree(root.left_, data,
                                                   point_indices_sorted[0:median_point_index],
                                                   axis, leaf_size)
            root.right_ = self.recursive_build_tree(root.right_, data,
                                                    point_indices_sorted[median_point_index:],
                                                    axis, leaf_size)
        return root

    @staticmethod
    def sort_key_by_val(key, value, sort_size):
        """ get pivot point from partly sorted points
            Args:
                points, points index and size of points to be sorted

            Returns:
                points and indices(partly sorted)
        """
        assert key.shape == value.shape
        sorted_idx = np.argsort(value[:sort_size])
        key[:sort_size] = key[sorted_idx]
        value[:sort_size] = value[sorted_idx]

    def get_partition(self, key, value):
        """ get median point along given axis, refer to get pivot method in quick sort
            Args:
                points and points index

            Returns:
                points, points index, median point index, median point value
        """
        assert key.shape == value.shape
        if key.shape[0] <= self.sample_size_:
            self.sort_key_by_val(key, value, key.shape[0])
            mid_idx = key.shape[0] // 2
            return key, value, mid_idx, value[mid_idx]
        else:
            self.sort_key_by_val(key, value, self.sample_size_)
            # 把排序好的sample数据中点移到最后，作为pivot点
            mid_idx = self.sample_size_ // 2
            key[mid_idx], key[-1] = key[-1], key[mid_idx]
            value[mid_idx], value[-1] = value[-1], value[mid_idx]
            pivot = value[-1]
            # 将小于pivot点的移动到左边，大于pivot点的移动到右面
            j = 0
            for i in range(value.shape[0]):
                if value[i] < pivot:
                    if i == j:
                        j += 1
                    else:
                        value[i], value[j] = value[j], value[i]
                        key[i], key[j] = key[j], key[i]
                        j += 1
            value[j], value[-1] = value[-1], value[j]
            key[j], key[-1] = key[-1], key[j]
            mid_point_index = j
            mid_point = value[j]
            return key, value, mid_point_index, mid_point

    @staticmethod
    def next_axis(data):
        # 按方差最大的轴进行分割，加快搜索速度
        a = np.var(data, axis=0)
        b = np.argmax(a)
        return np.argmax(np.var(data, axis=0))

    def query_neighbors(self, root: Node, data: np.ndarray,
                        result_set: KNNResultSet, query: np.ndarray, knn_num=5):
        if root is None:
            return

        if root.is_leaf():
            leaf_points = data[root.point_indices_, :]
            diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
            for i in range(diff.shape[0]):
                result_set.add_point(diff[i], root.point_indices_[i])
            return

        if query[root.axis_] < root.value_:
            self.query_neighbors(root.left_, data, result_set, query, knn_num)
            if math.fabs(query[root.axis_] - root.value_) < result_set.worst_dist_:
                self.query_neighbors(root.right_, data, result_set, query, knn_num)
        else:
            self.query_neighbors(root.right_, data, result_set, query, knn_num)
            if math.fabs(query[root.axis_] - root.value_) < result_set.worst_dist_:
                self.query_neighbors(root.left_, data, result_set, query, knn_num)
        return

    def query_radius(self, root: Node, data: np.ndarray,
                     result_set: RadiusNNResultSet, query: np.ndarray, radius=0.5):
        if root is None:
            return

        if root.is_leaf():
            leaf_points = data[root.point_indices_, :]
            diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
            for i in range(diff.shape[0]):
                result_set.add_point(diff[i], root.point_indices_[i])
            return

        if query[root.axis_] < root.value_:
            self.query_radius(root.left_, data, result_set, query, radius)
            if math.fabs(query[root.axis_] - root.value_) < result_set.worst_dist_:
                self.query_radius(root.right_, data, result_set, query, radius)
        else:
            self.query_radius(root.right_, data, result_set, query, radius)
            if math.fabs(query[root.axis_] - root.value_) < result_set.worst_dist_:
                self.query_radius(root.left_, data, result_set, query, radius)
        return

    def traverse_kdtree(self, root: Node, depth, max_depth):
        depth[0] += 1
        if max_depth[0] < depth[0]:
            max_depth[0] = depth[0]

        if root.is_leaf():
            # print(root)
            pass
        else:
            self.traverse_kdtree(root.left_, depth, max_depth)
            self.traverse_kdtree(root.right_, depth, max_depth)

        depth[0] -= 1


if __name__ == '__main__':
    abs_path = os.path.abspath(os.path.dirname(__file__))  # 数据集路径 (数据集直接放在当前路径下)
    filename = os.path.join(abs_path, '000000.bin')  # 只读取一个文件,如果要读取所有文件,需要循环读入
    # 读取数据并进行采样
    db_np_origin = read_velodyne_bin(filename)  # N*3
    db_np_idx = np.random.choice(db_np_origin.shape[0], size=(30000,))  # 随机采样30000个点
    db_np = db_np_origin[db_np_idx]

    # configuration
    dim = 3
    leaf_size = 20
    k = 5
    radius = 0.2
    # 建树
    tree = KDTree_opt(db_np, leaf_size, dim)
    # print(tree.root_)
    depth = [0]
    max_depth = [0]
    tree.traverse_kdtree(tree.root_, depth, max_depth)
    print("opt kdtree max depth: %d" % max_depth[0])

    # KNN search
    print("KNN search:")
    query = db_np[0, :]
    result_set = KNNResultSet(capacity=k)
    tree.query_neighbors(tree.root_, db_np, result_set, query)
    print(result_set)

    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    print(nn_idx[0:k])
    print(nn_dist[0:k])

    # radius NN search
    print("Radius search:")
    result_set = RadiusNNResultSet(radius=radius)
    tree.query_radius(tree.root_, db_np, result_set, query)
    print(result_set)
