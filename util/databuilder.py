# -*- coding:utf-8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import itertools
import os

import numpy as np
from scipy.sparse import csr_matrix

from util.matrix import Matrix


class DataBuilder:
    def __init__(self, file_name, k_folds=10):
        self.file_name = file_name
        self.k_folds = k_folds

    def read_ratings(self):
        with open(os.path.expanduser(self.file_name)) as f:
            raw_ratings = [self.parse_line(line) for line in itertools.islice(f, 0, None)]
        return raw_ratings

    def parse_line(self, line):
        line = line.split("\t")
        uid, iid, r, timestamp = (line[i].strip() for i in range(4))
        return uid, iid, float(r), timestamp

    def cv(self):
        raw_ratings = self.read_ratings()
        stop = 0
        raw_len = len(raw_ratings)
        offset = raw_len // self.k_folds
        left = raw_len % self.k_folds
        for fold_i in range(self.k_folds):
            start = stop
            stop += offset
            if fold_i < left:
                stop += 1
            yield self.mapping(raw_ratings[:start] + raw_ratings[stop:]), raw_ratings[start:stop]

    def mapping(self, raw_train_ratings):
        uid_dict = {}
        iid_dict = {}
        current_u_index = 0
        current_i_index = 0

        row = []
        col = []
        data = []
        for urid, irid, r, timestamp in raw_train_ratings:
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

        return Matrix(sparse_matrix, uid_dict, iid_dict)

    # def cv(self):
    #     sparse_matrix = self.mapping()
    #     print(sparse_matrix.shape)
    #     row_index = np.arange(sparse_matrix.shape[0])
    #     np.random.shuffle(row_index)
    #
    #     offset = row_index.size // self.k_folds
    #     left = row_index.size % self.k_folds
    #     stop = 0
    #
    #     for fold_i in range(self.k_folds):
    #         start = stop
    #         stop += offset
    #         if fold_i < left:
    #             stop += 1
    #         yield Matrix(sparse_matrix[np.append(row_index[:start], row_index[stop:], axis=0)]), \
    #               Matrix(sparse_matrix[row_index[start:stop]])

    def rmse(self, algorithm, pause=True):
        rmse_result = []

        for train_dataset, test_dataset in self.cv():
            algorithm.train(train_dataset)
            rmse_result.append(algorithm.estimate(test_dataset))
            if pause:
                break
        print(np.mean(rmse_result))
