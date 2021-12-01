import numpy as np
from sklearn.neighbors import KDTree


class OwnDBSCAN():
    def __init__(self, radius=0.4, min_samples=5):
        self.search_radius_ = radius
        self.min_samples_needed_ = min_samples
        self.labels_ = []

    def fit(self, data):
        N, _ = data.shape
        cls = -1 * np.ones(N, dtype=int)
        label = -1
        visited = np.zeros(N)
        search_index = list(range(N))
        # 优先搜索队列，先沿着一个中心点和簇去扩展中心点和边界点
        priority_search = []
        tree = KDTree(data)
        while search_index:
            if not priority_search:
                i = search_index.pop()
                if visited[i] == 1:
                    continue
                else:
                    visited[i] = 1
                nn_indices = tree.query_radius([data[i]], self.search_radius_)
                nn_indices = nn_indices[0]
                if nn_indices.shape[0] - 1 < self.min_samples_needed_:
                    cls[i] = -1
                else:
                    label += 1
                    cls[i] = label
                    priority_search.extend(nn_indices)
            else:
                i = priority_search.pop()
                if visited[i] == 1:
                    continue
                else:
                    visited[i] = 1
                    cls[i] = label
                nn_indices = tree.query_radius([data[i]], self.search_radius_)
                nn_indices = nn_indices[0]
                if nn_indices.shape[0] - 1 < self.min_samples_needed_:
                    pass
                else:
                    priority_search.extend(nn_indices)
                    priority_search = list(np.unique(priority_search))
        self.labels_ = cls


if __name__ == '__main__':
    # 单独跑本模块，根据提供的数据验证方法的正确性
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    dbscan = OwnDBSCAN(radius=5, min_samples=2)
    dbscan.fit(x)
    print(dbscan.labels_)
