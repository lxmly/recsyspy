
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import itertools
from scipy.sparse import csr_matrix
import numpy as np
from svdpp import SVDpp
from slop_one import SlopOne
from itemCF import ItemCF
from collections import defaultdict
from als import ExplicitALS
from matrix import Matrix

class DataBuilder:
    def __init__(self, file_name, k = 10):
        self.file_name = file_name
        self.k = k

    def read_ratings(self):
        with open(os.path.expanduser(self.file_name)) as f:
            raw_ratings = [self.parse_line(line) for line in itertools.islice(f, 0, None)]
        return raw_ratings

    def parse_line(self, line):
        line = line.split("\t")
        uid, iid, r, timestamp = (line[i].strip() for i in range(4))
        return uid, iid, float(r), timestamp

    # def k_folds(self, n_folds):
    #     seq = self.build_dataset()
    #     start = 0
    #     stop = 0
    #     seq_len = len(seq)
    #     offset = seq_len // n_folds
    #     left = seq_len % n_folds
    #     for fold_i in range(n_folds):
    #         start = stop
    #         stop += offset
    #         if fold_i < left:
    #             stop += 1
    #         yield seq[:start] + start[stop:], seq[start:stop]

    def k_folds(self):
        raw_dataset = self.read_ratings()
        uid_dict = {}
        iid_dict = {}
        current_u_index = 0
        current_i_index = 0

        row = []
        col = []
        data = []
        for urid, irid, r, timestamp in raw_dataset:
            try:
                uid = uid_dict[urid]
            except KeyError:
                uid = current_u_index
                uid_dict[urid] = current_u_index
                current_u_index += 1
            try:
                iid = iid_dict[irid]
            except KeyError:
                iid = current_i_index
                iid_dict[irid] = current_i_index
                current_i_index += 1

            row.append(uid)
            col.append(iid)
            data.append(r)
        sparse_matrix = csr_matrix((data, (row, col)))
        print(sparse_matrix.shape)
        row_index = np.arange(sparse_matrix.shape[0])
        np.random.shuffle(row_index)

        offset = row_index.size // self.k
        left = row_index.size % self.k
        stop = 0

        for fold_i in range(self.k):
            start = stop
            stop += offset
            if fold_i < left:
                stop += 1
            yield Matrix(sparse_matrix[np.append(row_index[:start], row_index[stop:], axis=0)]), \
                  Matrix(sparse_matrix[row_index[start:stop]])

    def rmse(self, algorithm, pause=True):
        rmse_result = []

        for train_dataset, test_dataset in self.k_folds():
            algorithm.train(train_dataset)
            rmse_result.append(algorithm.estimate(test_dataset))
            if pause:
                break
        print(np.mean(rmse_result))





if __name__ == '__main__':
    file_name = '/Users/fanruiqiang/work/data/ml-100k/u.data'
    data_builder = DataBuilder(file_name, 7)
    #print([x for x in train_dataset.get_users()])
    #slopOne = SlopOne(False)
    #slopOne.train(train_dataset)
    #print(slopOne.estimate(test_dataset))
    # itemCF = ItemCF()
    # data_builder.rmse(itemCF)
    als = ExplicitALS()
    data_builder.rmse(als)
    #print(train_dataset.shape, test_dataset.shape)
    #slopOne.train(coo_matrix, uid_dict, iid_dict)
    #print(slopOne.estimate(0,7))
    #row = np.array([0, 3, 1, 0])
    #col = np.array([0, 3, 1, 2])
    #data = np.array([4, 5, 7, 9])
    #matrix = csc_matrix((data, (row, col)))
    #print(matrix.toarray())

    # for train_dataset, test_dataset in data_builder.k_folds():
    #     counts = load_matrix(train_dataset)
    #     implicitMF = ImplicitMF(counts)
    #     implicitMF.train_model()