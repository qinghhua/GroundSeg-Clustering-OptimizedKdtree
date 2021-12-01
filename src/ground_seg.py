import numpy as np
import random
import math


class RANSAC:
    def __init__(self, eps=0.5, max_iter=500, inliers_ratio=0.5, n=3):
        self.eps_ = eps
        self.max_iter_ = max_iter
        self.inliers_ratio_ = inliers_ratio
        self.n_ = n

    @staticmethod
    def is_degenerate(maybe_inliers, min_dist):
        degenerate = False
        v1 = maybe_inliers[0, :] - maybe_inliers[1, :]
        v2 = maybe_inliers[1, :] - maybe_inliers[2, :]
        v3 = maybe_inliers[0, :] - maybe_inliers[2, :]
        # 三点共线，叉乘结果为0向量
        if np.all(np.cross(v1, v2) == 0):
            degenerate = True
        # 三点距离太近
        if (np.linalg.norm(v1) < min_dist and
                np.linalg.norm(v2) < min_dist and
                np.linalg.norm(v3) < min_dist):
            degenerate = True
        return degenerate

    def get_maybe_inliers(self, data):
        index = random.sample(range(1, data.shape[0]), self.n_)
        maybe_inliers = []
        for i in range(len(index)):
            maybe_inliers.append(data[index[i], :])
        maybe_inliers = np.array(maybe_inliers)
        return maybe_inliers

    def get_plane_coef(self, maybe_inliers, normalized=True):
        # '''法向量估计， 一组点的协防差矩阵的最小特征值对应的特征向量是法向量
        #     已知3个平面点point, 求平面方程，返回系数矩阵
        #     Args:
        #         point: 3*3 array
        #         normalize: bool
        #
        #     Returns:
        #         coef:  1*4 array
        #     '''
        # # 检查是否共线
        #
        # # ax+by+cz+d = 0
        inliers = np.asarray(maybe_inliers)
        data_mean = np.mean(inliers, axis=0)
        data_norm = inliers - data_mean
        data_norm = data_norm.T  # 3 * 3, 一条数据为一个列向量
        cov_matrix = np.dot(data_norm, data_norm.T) / data_norm.shape[1]
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
        normal = np.zeros((self.n_ + 1))
        normal[0:3] = eigenvectors[:, 2].reshape(3, )  # 最小特征值对应的特征向量为法向量
        if normalized:  # 归一化
            rms = math.sqrt(normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2)
            normal[0] = normal[0] / rms
            normal[1] = normal[1] / rms
            normal[2] = normal[2] / rms
        normal[3] = -normal[0] * inliers[0, 0] - \
                    normal[1] * inliers[0, 1] - \
                    normal[2] * inliers[0, 2]
        # # 法向量计算： a  = (y2-y1)*(z3-z1) - (y3-y1)*(z2-z1)
        # #            b = (z2-z1)*(x3-x1) - (z3-z1)*(x2-x1)
        # #            c = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        # #            d = -a*x1 - b*y1 - c*z1
        # maybe_inliers = np.asarray(maybe_inliers)
        # v21 = maybe_inliers[1, :] - maybe_inliers[0, :]
        # v31 = maybe_inliers[2, :] - maybe_inliers[0, :]
        # a = (v21[1] * v31[2]) - (v31[1] * v21[2])
        # b = (v21[2] * v31[0]) - (v31[2] * v21[0])
        # c = (v21[0] * v31[1]) - (v31[0] * v21[1])
        # if True:
        #     r = math.sqrt(a ** 2 + b ** 2 + c ** 2)
        #     a = a / r
        #     b = b / r
        #     c = c / r
        # d = - a * maybe_inliers[0, 0] - b * maybe_inliers[0, 1] - c * maybe_inliers[0, 2]
        # normal = np.array([a, b, c, d])
        return normal

    def get_model(self, data):
        N, _ = data.shape
        max_points = 0
        normal = np.asarray([0, 0, 1]).reshape(1, -1)
        for iter in range(self.max_iter_):
            maybe_inliers = self.get_maybe_inliers(data)
            if self.is_degenerate(maybe_inliers, 0.1):
                continue
            coef = self.get_plane_coef(maybe_inliers).reshape(-1, 1)
            if abs(np.dot(normal, coef[:3])) > math.cos(10 / 180 * math.pi):
                dists = (np.abs(np.dot(data, coef[:3, :]) + coef[3, :])).reshape(-1, )
                plane_points_idx = np.where(dists < self.eps_)
                T = np.shape(plane_points_idx)[1]
                if T > max_points:
                    max_points = T
                    if np.dot(normal, coef[:3]) > 0.0:
                        model = coef
                    else:
                        model = -1.0 * coef
                    ground_point_idx = plane_points_idx
                    non_ground_point_idx = np.where(dists >= self.eps_)
                if max_points > self.inliers_ratio_ * N:
                    break
        print('origin data points num:', N)
        print('segmented data points num:', np.shape(non_ground_point_idx)[1])
        print("plane normal:", model)
        print("iter num :", iter)
        return ground_point_idx, non_ground_point_idx


if __name__ == '__main__':
    ransac = RANSAC()
