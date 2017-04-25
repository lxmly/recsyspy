# -*- coding:utf-8 -*-

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import itertools


class SlopOne:
    def __init__(self):
        pass
    def train(self, train_dataset):
        item_num = train_dataset.shape[1]

        freq = lil_matrix((item_num, item_num),  dtype=np.int8)
        self.dev = lil_matrix((item_num, item_num),  dtype=np.double)

        print("total {} user".format(train_dataset.shape[0]))
        m = 0

        for u, ii, rr in train_dataset.get_users():
            m += 1
            if m%50 == 0:
                print("current {}th".format(m))
            for k in range(ii.size):
                i, j = ii[k], ii[k + 1]
                if i > j:
                    i, j = j, i
                freq[i, j] += 1
                self.dev[i, j] += rr[i] - rr[j]

        nonzero_indices = self.dev.nonzero()

        self.dev[nonzero_indices] /= freq[nonzero_indices]

        nonzero_indices_T = self.dev.transpose().nonzero()

        self.dev[nonzero_indices_T] = self.dev[nonzero_indices]

        # for i,j in zip(dev.nonzero()):
        #     if i > j:
        #        i, j = j, i
        #     dev[i, j] /= freq[i, j]

        self.user_mean = train_dataset.get_users_mean()

    def estimate(self, test_dataset):
        est = [self.predict(u, i, test_dataset) - r
               for u, i, r in zip(test_dataset.row, test_dataset.col, test_dataset.data)]

        return np.sqrt(np.mean(est ** 2))

    def predict(self, u, k, test_dataset):
        ii, rr = test_dataset.get_user(u)
        for i in ii if dev.contain_uv(k, i):


        Ru = np.unique(test_dataset.getrow(u).tocoo().col)
        return self.user_mean[u] + np.sum([self.dev[i, j] for j in Ru]) / len(Ru)








