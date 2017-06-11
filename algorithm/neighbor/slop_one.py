# -*- coding:utf-8 -*-

from __future__ import division, print_function

import numpy as np
from scipy.sparse import lil_matrix

from algorithm.mf.estimator import Estimator


class SlopOne(Estimator):
    """
    属性
    ---------
    is_weighted : slopOne or weightedSlopOne
    """
    def __init__(self, is_weighted=False):
        self.is_weighted = is_weighted

    def _train(self):
        item_num = self.train_dataset.matrix.shape[1]

        self.freq = lil_matrix((item_num, item_num),  dtype=np.int8)
        self.dev = lil_matrix((item_num, item_num),  dtype=np.double)
        user_num = self.train_dataset.matrix.shape[0]
        cur = 0
        for u, (ii, rr) in self.train_dataset.get_users():
            cur += 1
            for k in range(len(ii) - 1):
                k1, k2 = k, k+1
                i1, i2 = ii[k1], ii[k2]
                if i1 > i2:
                    i1, i2 = i2, i1
                    k1, k2 = k2, k1
                self.freq[i1, i2] += 1
                self.dev[i1, i2] += rr[k1] - rr[k2]
            self.progress(cur, user_num, 50)

        nonzero_indices = self.freq.nonzero()
        self.dev[nonzero_indices] /= self.freq[nonzero_indices]

        self.dev[(nonzero_indices[1], nonzero_indices[0])] = -self.dev[nonzero_indices]
        self.freq[(nonzero_indices[1], nonzero_indices[0])] = self.freq[nonzero_indices]

        # for i,j in zip(dev.nonzero()):
        #     if i > j:
        #        i, j = j, i
        #     dev[i, j] /= freq[i, j]

        self.dev = self.dev.A
        self.freq = self.freq.A
        self.user_means = self.train_dataset.get_user_means()
        self.ratings = self.train_dataset.matrix.A

    def predict(self, u, i):
        N = [j for j in self.train_dataset.get_user(u)[0] if self.freq[i, j] > 0]
        est = self.user_means[u]

        if N:
            if self.is_weighted:
                est = sum([(self.ratings[u, j] + self.dev[i, j]) * self.freq[i, j] for j in N]) /\
                      sum([self.freq[i, j] for j in N])
            else:
                est += np.mean([self.dev[i, j] for j in N])
        return est








