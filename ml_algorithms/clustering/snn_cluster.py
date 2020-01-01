# Author: zhuguowei  <csuzhuge16@163.com>
'''
SNN Cluster Algorithms for Spatial Data Mining V0.1
'''
import sys,os
import time
import pandas as pd
import numpy as np
import argparse
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
def time_me(fn):
    def _wrapper(*args, **kwargs):
        start = time.clock()
        res = fn(*args, **kwargs)
        print("%s cost %s second"%(fn.__name__, time.clock() - start))
        return res
    return _wrapper

class SNNCluster(object):

    def __init__(self, k, min_pts, eps, input_data='data/text.csv', output='result'):
        """

        Returns:
            object:
        """
        self.k = k  # 共享近邻数k
        self.min_pts = min_pts  # 最少点数
        self.eps = eps  #
        self.input_data = input_data  # 输入数据
        self.point_counts = 0  # 点数
        self.points_df = None
        self.snn_graph = None
        self.snn_density = None
        self.core_points_list = None  # step4 find the core points
        self.output = output
        self.core_or_not = None
        self.core_neighbors = []
        self.visited = []  # list to store points visited
        self.labels = []
        self.colors = []
        self.tmp_neighbor_core =None
        pass

    @time_me
    def read_data(self):
        '''
         读取数据
        Args:
            file: str
        Returns:
        '''
        self.points_df = pd.read_csv(self.input_data, header=0)
        self.point_counts = len(self.points_df["ID"])

    @time_me
    def get_points(self):
        '''
        获取点
        Returns:

        '''
        return self.points_df

    @time_me
    def _distance_matrix(self, points=None):
        '''
        计算距离矩阵
        Args:
            points:
        Returns:
            list(list())
        '''
        if not points:
            points = self.points_df[["X", "Y"]]
        self.distance_matrix = pdist(points, metric='euclidean')
        self.distance_matrix = squareform(self.distance_matrix)
        return self.distance_matrix

    @time_me
    def _knn_list(self, distance_mat=None, k=None):
        '''
        根据距离矩阵和k参数计算KNN列表
        Returns:

        '''
        n = 0
        if not distance_mat:
            distance_mat = self.distance_matrix
            n = self.point_counts
        if not k:
            k = self.k
        self.similarity_mat = np.zeros((n, k))
        self.similarity_mat = np.argsort(distance_mat)[:, :k + 1]
        return self.similarity_mat

    def _count_intersection(self, list1, list2):
        '''
        求交集数量
        Args:
            list1:
            list2:

        Returns:

        '''
        intersection = 0
        for i in list1:
            if i in list2:
                intersection = intersection + 1
        return intersection
    @time_me
    def construct_snn_graph(self, similarity_matrix=None, k=None):
        '''
        稀疏化相似矩阵，构造SNN图. ***可优化
        Args:
            similarity_matrix:
            k:
        Returns:

        '''
        point_count = 0
        if not k:
            k = self.k
        if similarity_matrix is None:
            similarity_matrix = self.similarity_mat
            print("points count: ", self.point_counts)
            point_count = self.point_counts
        else:
            point_count = len(similarity_matrix)
        # self.snn_graph = [[0 for i in range(point_count)] for j in range(point_count)]
        self.snn_graph = np.zeros((point_count, point_count))
        for i in range(0, point_count - 1):
            for j in range(i + 1, point_count):
                if j in similarity_matrix[i] and i in similarity_matrix[j]:
                    count = self._count_intersection(similarity_matrix[i], similarity_matrix[j])
                    self.snn_graph[i][j] = count
                    # self.snn_graph[j][i] = self.snn_graph[i][j]
        self.snn_graph = self.snn_graph.T + self.snn_graph
        return self.snn_graph

    def _find_snn_dense_of_point(self, lst):
        num = 0
        for i in range(0, self.point_counts):
            if lst[i] >= self.eps:
                num = num + 1
        return num

    @time_me
    def _find_snn_dense_of_points(self, snn_graph=None):
        '''

        Returns:

        '''
        if snn_graph is None:
            snn_graph = self.snn_graph
        self.snn_density = [None for i in range(len(snn_graph))]
        for i in range(len(snn_graph)):
            self.snn_density[i] = self._find_snn_dense_of_point(snn_graph[i])
        return self.snn_density

    def is_core_point(self, point) -> bool:
        if point >= self.min_pts:
            return True
        else:
            return False

    @time_me
    def core(self, x, y):
        if x >= self.min_pts:
            return y
        else:
            return None

    @time_me
    def _find_core_points(self):
        self.core_or_not = [False for i in range(len(self.snn_density))]
        for i in range(len(self.snn_density)):
            self.core_or_not[i] = self.is_core_point(self.snn_density[i])
        core_points_list1 = []
        snn_density2 = zip(self.snn_density, [i for i in range(len(self.snn_density))])
        snn_density2 = list(snn_density2)
        for i in range(len(snn_density2)):
            core_points_list1.append(self.core(snn_density2[i][0], snn_density2[i][1]))
        self.core_points_list = [x for x in core_points_list1 if x != None]
        return self.core_points_list

    @time_me
    def _find_core_neighbors(self, p):
        tmp_core_neighbors = []
        p2 = None
        for i in range(0, len(self.core_points_list)):
            p2 = self.core_points_list[i]
            if p != p2 and self.snn_graph[p][p2] >= self.eps:
                tmp_core_neighbors.append(p2)
        # print("inside core_neighbor: ", tmp_core_neighbors)
        self.tmp_neighbor_core = tmp_core_neighbors

        return tmp_core_neighbors

    @time_me
    def expand_cluster(self, C):
        while len(self.core_neighbors) > 0:
            p = self.core_neighbors.pop(0)
            if p in self.visited:
                continue
            self.labels[p] = C
            self.visited.append(p)
            self._find_core_neighbors(p)
            self.core_neighbors.extend(self.tmp_neighbor_core)
        return self.labels


    @time_me
    def find_cluster_from_core_points(self):
        self.visited = []
        self.labels = [0 for i in range(self.point_counts)]
        c = 0
        for i in range(len(self.core_points_list)):
            p = self.core_points_list[i]
            if p in self.visited:
                continue
            self.visited.append(p)
            c = c + 1
            self.labels[p] = c
            self.core_neighbors = self._find_core_neighbors(p=p) # why return None?
            self.expand_cluster(c)
        # print("labels find_cluster_from_core_points：",self.labels)
        return self.labels


    def _determine_final_cluster(self):
        for i in range(self.point_counts):
            not_noise = False
            max_similarity = -sys.maxsize
            best_core = -1
            sim = None
            if (self.core_or_not[i]):
                continue
            for j in range(len(self.core_points_list)):
                p = self.core_points_list[j]
                sim = self.snn_graph[i][p]
                if (sim >= self.eps):
                    not_noise = True
                else:
                    self.labels[i] = 0
                    break
                if (sim > max_similarity):
                    max_similarity = sim
                    best_core = p
            if not_noise:
                self.labels[i] = self.labels[best_core]
        number_of_clusters = max(self.labels)
        print('number_of_clusters: ', number_of_clusters)

        name1 = str(os.path.basename(self.input_data).replace('.csv', '')) + '_k_' + str(self.k) + '_eps_' + str(self.eps) + '_minpts_' + str(
            self.min_pts)
        name = name1 + '.txt'
        self._write_result(name=name)
        return self.labels

    def _write_result(self, name):
        '''
        将聚类结果（cluster）写入文件
        :param path:
        :return:
        '''
        if not os.path.exists(self.output):
            os.mkdir(self.output)
        save_path = os.path.join(self.output,name)
        outfile = open(save_path, 'w')
        for i in range(len(self.labels)):
            outfile.write(str(self.labels[i]))
            outfile.write('\n')
        outfile.close()

    @time_me
    def _visualize(self):
        colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf',
                   '#a65628', '#984ea3', '#999999', '#e41a1c',
                   '#dede00']
        for i in range(self.point_counts):
            plt.scatter(self.points_df["X"][i], self.points_df["Y"][i], color=colors[self.labels[i]])
        title = '_'.join(['neighbor', str(self.k), 'minpts', str(self.min_pts), 'eps', str(self.eps)])
        plt.title("SNN Clustering-"+title)
        plt.xlabel("X")
        plt.ylabel("Y")
        # plt.legend()
        plt.show()

    def get_core_points(self):
        return self.core_points_list

    def run(self):
        '''
        执行聚类过程
        :return:
        '''
        self.read_data()
        self._distance_matrix()
        self._knn_list()
        self.construct_snn_graph()
        # Step3: Find the SNN density of each point
        self._find_snn_dense_of_points()
        # Step4: Find the core points
        self._find_core_points()
        # Step5: Find clusters from the core points
        self.find_cluster_from_core_points()
        # print("cluster: ",self.find_cluster_from_core_points())
        self._determine_final_cluster()
        self._visualize()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default='data/text.csv', type=str, help="input data for clustering")
    parser.add_argument("--k", default=20, type=int, help="share nearest neighbor k. default is 20")
    parser.add_argument("--min_pts", default=13, type=int, help="minimized points to determine the core points. default is 13")
    parser.add_argument("--eps", default=6, type=int, help="radius threhold to determine the SNN similarity. default is 6")
    parser.add_argument("--output", default='result/', type=str, help="the file path to store the result")
    args = parser.parse_args()
    snn = SNNCluster(input_data=args.input, k=args.k, min_pts=args.min_pts, eps=args.eps, output=args.output)
    snn.run()

