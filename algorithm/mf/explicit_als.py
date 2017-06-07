# -*- coding:utf-8 -*-

import numpy as np
import scipy.sparse as sparse
from algorithm.mf.estimator import IterationEstimator


class ExplicitALS(IterationEstimator):
    """显式交替最小二乘，算法表现一般，从它的损失函数也可以看出，
       是最简单的svd。只不过ALS相比SGD速度快一点, 一般10次迭代就能收敛
    """

    def __init__(self, n_factors=20, n_epochs=10, reg=0.1):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg

    #交替！
    def alternative(self, X, Y, is_user):
        reg_I = self.reg * sparse.eye(self.n_factors)
        uids = self.train_dataset.uids if is_user else self.train_dataset.iids

        for u in uids:
            if is_user:
                action_idx = self.train_dataset.get_user(u)[0]
            else:
                action_idx = self.train_dataset.get_item(u)[0]
            Y_u = Y[action_idx]

            if is_user:
                ru = self.train_dataset.matrix.A[u, action_idx]
            else:
                ru = self.train_dataset.matrix.A[action_idx, u].T

            X[u] = np.linalg.solve(np.dot(np.transpose(Y_u), Y_u) + reg_I, np.dot(Y_u.T, ru))

    def _prepare(self, train_dataset):
        self.train_dataset = train_dataset
        self.user_num = self.train_dataset.matrix.shape[0]
        self.item_num = train_dataset.matrix.shape[1]
        self.X = np.random.normal(size=(self.user_num, self.n_factors))
        self.Y = np.random.normal(size=(self.item_num, self.n_factors))

    def _iteration(self):
        self.alternative(self.X, self.Y, True)
        self.alternative(self.Y, self.X, False)

    def _pred(self):
        return np.dot(self.X, self.Y.T)

    def predict(self, u, i, r):
        est = np.dot(self.X[u,:], self.Y[i,:])
        return r, est