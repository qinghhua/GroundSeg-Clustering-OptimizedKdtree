import copy


class DistIndex:
    def __init__(self, distance, index):
        self.dist_ = distance
        self.index_ = index

    def __lt__(self, other):
        return self.dist_ < other.dist_


class KNNResultSet:
    def __init__(self, capacity):
        self.capacity_ = capacity
        self.count_ = 0
        self.worst_dist_ = 1e10
        self.dist_index_list_ = []
        for i in range(capacity):
            self.dist_index_list_.append(DistIndex(self.worst_dist_, 0))

        self.comparison_counter_ = 0

    def size(self):
        return self.count_

    def worstDist(self):
        return self.worst_dist_

    def add_point(self, dist, index):
        # 向result set中添加元素，并更新最小距离
        self.comparison_counter_ += 1
        if dist > self.worst_dist_:
            return

        if self.count_ < self.capacity_:
            self.count_ += 1

        i = self.count_ - 2
        while i >= 0:
            if dist < self.dist_index_list_[i].dist_:
                self.dist_index_list_[i + 1] = copy.deepcopy(self.dist_index_list_[i])
                i -= 1
            else:
                break

        self.dist_index_list_[i + 1].dist_ = dist
        self.dist_index_list_[i + 1].index_ = index
        self.worst_dist_ = self.dist_index_list_[self.capacity_ - 1].dist_

    def __str__(self):
        output = ''
        for i, dist_index in enumerate(self.dist_index_list_):
            output += '%d - %.2f\n' % (dist_index.index_, dist_index.dist_)
        output += 'In total %d comparison operations.' % self.comparison_counter_
        return output


class RadiusNNResultSet:
    def __init__(self, radius):
        self.radius_ = radius
        self.count_ = 0
        self.worst_dist_ = radius
        self.dist_index_list_ = []

        self.comparison_counter_ = 0

    def size(self):
        return self.count_

    def worstDist(self):
        return self.radius_

    def add_point(self, dist, index):
        self.comparison_counter_ += 1
        if dist > self.worst_dist_:
            return

        self.count_ += 1
        self.dist_index_list_.append(DistIndex(dist, index))

    def __str__(self):
        self.dist_index_list_.sort()
        output = ''
        for i, dist_index in enumerate(self.dist_index_list_):
            output += '%d - %.2f\n' % (dist_index.index_, dist_index.dist_)
        output += 'In total %d neighbors within %f.\nThere are %d comparison operations.' \
                  % (self.count_, self.radius_, self.comparison_counter_)
        return output
