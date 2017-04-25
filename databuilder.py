
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import itertools
from scipy.sparse import csc_matrix
import numpy as np
from svdpp import SVDpp
from slop_one import SlopOne
from collections import defaultdict

class DataBuilder:
    def __init__(self, file_name):
        self.file_name = file_name

    def read_ratings(self, file_name):
        with open(os.path.expanduser(file_name)) as f:
            raw_ratings = [self.parse_line(line) for line in itertools.islice(f, 0, None)]
        return raw_ratings
    def parse_line(self, line):
        line = line.split("\t")
        uid, iid, r, timestamp = (line[i].strip() for i in range(4))
        return uid, iid, float(r), timestamp

    def k_folds(self, seq, n_folds):
        if n_folds > len(seq) or n_folds < 2:
            raise ValueError("")
        start, stop = 0

        seq_len = len(seq)
        offset = seq_len // n_folds
        left = seq_len % n_folds
        for fold_i in range(n_folds):
            start = stop
            stop += offset
            if fold_i < left:
                stop += 1
            yield seq[:start] + start[stop:], seq[start:stop]


    def build_trainset(self):
        raw_trainset = self.read_ratings(self.file_name)
        uid_dict = {}
        iid_dict = {}
        current_u_index = 0
        current_i_index = 0

        row = []
        col = []
        data = []
        for urid, irid, r, timestamp in raw_trainset:
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
        sparse_matrix = csc_matrix((data, (row, col)))
        print(sparse_matrix.shape)
        row_index = np.arange(sparse_matrix.shape[0])
        np.random.shuffle(row_index)
        stop = int(row_index.size / 5)
        train, test = row_index[stop:], row_index[0:stop]
        train_dataset, test_dataset =  sparse_matrix[train, :], sparse_matrix[test, :]
        return Matrix(train_dataset), Matrix(test_dataset)

class Matrix:

    def __init__(self, sparse_matrix):
        self.matrix = sparse_matrix.tocoo()
        self._global_mean = None
        self.uids = set(self.matrix.row)
        self.iids = set(self.matrix.col)

    def get_item(self, i):
        return self.matrix.getcol(i).tocoo()

    def get_user(self, u):
        """ return user rating detail"""
        rating = self.matrix.getrow(u).tocoo()
        return rating.col, rating.data

    def get_users(self):
        for u in self.get_uids():
            yield u, self.get_user(u)

    def get_users_mean(self):
        """ compute the mean rating of each user """

        users_mean = {}
        for u in self.get_uids():
            users_mean[u] = np.mean(self.get_user(u)[1])
        return users_mean

    def all_ratings(self):
        """ return iterator(u,v,r)"""
        return itertools.izip(self.matrix.row, self.matrix.col, self.matrix.data)

    def get_uids(self):
        return np.unique(self.matrix.row)

    def get_iids(self):
        return np.unique(self.matrix.col)

    def cotain_ui(self, u, i):
        return u in self.uids and i in self.iids

    @property
    def global_mean(self):
        if self._global_mean is None:
            self._global_mean = np.sum(self.matrix.data) / self.matrix.nnz
        return self._global_mean

if __name__ == '__main__':
    file_name = '/Users/fanruiqiang/work/data/ml-100k/u.data'
    data_builder = DataBuilder(file_name)
    train_dataset, test_dataset = data_builder.build_trainset()
    print(train_dataset.all_ratings)
    #slopOne = SlopOne()
    #slopOne.train(coo_matrix, uid_dict, iid_dict)
    #print(slopOne.estimate(0,7))
    #row = np.array([0, 3, 1, 0])
    #col = np.array([0, 3, 1, 2])
    #data = np.array([4, 5, 7, 9])
    #matrix = csc_matrix((data, (row, col)))
    #print(matrix.toarray())