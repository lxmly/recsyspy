# -*- coding:utf-8 -*-

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix
import itertools
from sklearn.metrics import mean_squared_error

class SlopOne():
    def __init__(self):
        pass
    def train(self, train_dataset, uid_dict, iid_dict):
        item_num = train_dataset.shape[1]

        freq = lil_matrix((item_num, item_num),  dtype=np.int8)
        self.dev = lil_matrix((item_num, item_num),  dtype=np.double)

        print("total {} user".format(train_dataset.shape[0]))
        m = 0

        for u in np.unique(train_dataset.row):
            m += 1
            if m%50 == 0:
                print("current {}th".format(m))
            ratings = train_dataset.getrow(u)
            ilist = ratings.tocoo().col
            for k in range(len(ilist) - 1):
                i,j = ilist[k], ilist[k + 1]
                if i > j:
                    i, j = j, i
                freq[i, j] += 1
                self.dev[i, j] += ratings[0, i] - ratings[0, j]

        nonzero_indices = self.dev.nonzero()

        self.dev[nonzero_indices] /= freq[nonzero_indices]

        nonzero_indices_T = self.dev.transpose().nonzero()

        self.dev[nonzero_indices_T] = self.dev[nonzero_indices]

        # for i,j in zip(dev.nonzero()):
        #     if i > j:
        #        i, j = j, i
        #     dev[i, j] /= freq[i, j]

        self.user_mean = [np.mean(train_dataset.getrow(u).data) for u in np.unique(train_dataset.row)]
        self.coo_matrix = train_dataset
        self.uid_dict = uid_dict
        self.iid_dict = iid_dict

    def estimate(self, test_dataset):
        est = [self.predict(u,i, test_dataset) - r
               for u, i, r in zip(test_dataset.row, test_dataset.col, test_dataset.data)]
        return np.sqrt(np.mean(est ** 2))

    def predict(self, u, i, test_dataset):
        Ru = np.unique(test_dataset.getrow(u).tocoo().col)
        return self.user_mean[u] + np.sum([self.dev[i, j] for j in Ru]) / len(Ru)








