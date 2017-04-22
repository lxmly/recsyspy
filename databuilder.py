from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import itertools
from scipy.sparse import csc_matrix
import numpy as np
from svdpp import SVDpp
from slop_one import SlopOne

class DataBuilder:
    def __init__(self, file_name):
        self.file_name = file_name

    def read_ratings(self, file_name):
        path = os.path.expanduser(file_name)
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
        return DataSet(uid_dict, iid_dict, csc_matrix((data, (row, col))))

class DataSet:
    def __init__(self, matrix, uid_dict, iid_dict):
        self.matrix = matrix
        self.uid_dict = uid_dict
        self.iid_dict = iid_dict


if __name__ == '__main__':
    file_name = 'E:/work/data/ml-100k/u1.test'
    data_builder = DataBuilder(file_name)
    uid_dict, iid_dict, coo_matrix = data_builder.build_trainset()
    slopOne = SlopOne()
    slopOne.train(coo_matrix, uid_dict, iid_dict)
    print(slopOne.estimate(0,7))
    #row = np.array([0, 3, 1, 0])
    #col = np.array([0, 3, 1, 2])
    #data = np.array([4, 5, 7, 9])
    #matrix = csc_matrix((data, (row, col)))
    #print(matrix.toarray())