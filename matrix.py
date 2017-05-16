
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import numpy as np
import itertools
import six

class Matrix:

    def __init__(self, sparse_matrix):
        self.matrix = sparse_matrix.todok()
        self._global_mean = None
        coo_matrix = sparse_matrix.tocoo()
        self.uids = set(coo_matrix.row)
        self.iids = set(coo_matrix.col)

    def get_item(self, i):
        rating = self.matrix.getcol(i).tocoo()
        return rating.row, rating.data

    def get_user(self, u):
        """ return user rating detail"""
        rating = self.matrix.getrow(u).tocoo()
        return rating.col, rating.data

    def get_users(self):
        for u in self.get_uids():
            yield u, self.get_user(u)

    def get_user_means(self):
        """ compute the mean rating of each user """
        users_mean = {}
        for u in self.get_uids():
            users_mean[u] = np.mean(self.get_user(u)[1])
        return users_mean

    def get_item_means(self):
        """ compute the mean rating of each user """
        item_means = {}
        for i in self.get_iids():
            item_means[i] = np.mean(self.get_item(i)[1])
        return item_means

    def all_ratings(self):
        coo_matrix = self.matrix.tocoo()
        return itertools.izip(coo_matrix.row, coo_matrix.col, coo_matrix.data)

    def get_uids(self):
        return np.unique(self.matrix.tocoo().row)

    def get_iids(self):
        return np.unique(self.matrix.tocoo().col)

    def has_user(self, u):
        return u in self.uids

    def has_item(self, i):
        return i in self.iids

    @property
    def global_mean(self):
        if self._global_mean is None:
            self._global_mean = np.sum(self.matrix.data) / self.matrix.size
        return self._global_mean