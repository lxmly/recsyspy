# -*- coding:utf-8 -*-

import numpy as np
from scipy.sparse import lil_matrix

from algorithm.mf.estimator import Estimator
from util.matrix import Matrix


class SlopOne(Estimator):

    def __init__(self, is_weighted=False):
        self.is_weighted = is_weighted

    def train(self, train_dataset):
        item_num = train_dataset.matrix.shape[1]

        freq = lil_matrix((item_num, item_num),  dtype=np.int8)
        dev = lil_matrix((item_num, item_num),  dtype=np.double)

        len = train_dataset.matrix.shape[0]
        m = 0
        for u, (ii, rr) in train_dataset.get_users():
            m += 1
            if m % 300 == 0:
                progress = 100 * (m / len)
                print("progress: %.2f%%" % progress)
            for k in range(ii.size - 1):
                k1, k2 = k, k+1
                i1, i2 = ii[k1], ii[k2]
                if i1 > i2:
                    i1, i2 = i2, i1
                    k1, k2 = k2, k1
                freq[i1, i2] += 1
                dev[i1, i2] += rr[k1] - rr[k2]

        nonzero_indices = freq.nonzero()

        dev[nonzero_indices] /= freq[nonzero_indices]

        dev[(nonzero_indices[1], nonzero_indices[0])] = -dev[nonzero_indices]
        freq[(nonzero_indices[1], nonzero_indices[0])] = freq[nonzero_indices]

        # for i,j in zip(dev.nonzero()):
        #     if i > j:
        #        i, j = j, i
        #     dev[i, j] /= freq[i, j]

        self.dev = Matrix(dev)
        self.freq = Matrix(freq)
        self.user_means = train_dataset.get_user_means()
        self.train_dataset = train_dataset

    def predict(self, u, i, r):
        N = [j for j in self.train_dataset.get_user(u)[0] if self.freq.matrix[i, j] > 0]
        est = self.user_means[u]

        if N:
            if self.is_weighted:
                est = sum([(self.train_dataset.matrix[u, j] + self.dev.matrix[i, j]) * self.freq.matrix[i, j] for j in N]) /\
                      sum([self.freq.matrix[i, j] for j in N])
            else:
                est += np.mean([self.dev.matrix[i, j] for j in N])
        return r, est








