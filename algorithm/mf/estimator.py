# -*- coding:utf-8 -*-

from __future__ import division, print_function

import numpy as np


class Estimator(object):
    """基础算法流程
    """

    def __init__(self):
        pass

    def train(self, train_dataset):
        raise NotImplementedError()

    def predict(self, u, i, r):
        raise NotImplementedError()

    def estimate(self, raw_test_dataset):
        users_mean = self.train_dataset.get_user_means()
        items_mean = self.train_dataset.get_item_means()

        l = len(raw_test_dataset)
        predictions = []
        m = 0
        alg_count = 0

        for raw_u, raw_i, r, _ in raw_test_dataset:
            m += 1
            has_raw_u = raw_u in self.train_dataset.uid_dict
            has_raw_i = raw_i in self.train_dataset.iid_dict

            if not has_raw_u and not has_raw_i:
                real, est = r, self.train_dataset.global_mean
            elif not has_raw_u:
                i = self.train_dataset.iid_dict[raw_i]
                real, est = r, items_mean[i]
            elif not has_raw_i:
                u = self.train_dataset.uid_dict[raw_u]
                real, est = r, users_mean[u]
            else:
                u = self.train_dataset.uid_dict[raw_u]
                i = self.train_dataset.iid_dict[raw_i]
                real, est = self.predict(u, i, r)
                alg_count += 1

            est = min(5, est)
            est = max(1, est)
            predictions.append((real - est) ** 2)
            if m%300 == 0:
                progress = 100 * (m / l)
                print("progress: {:.2f}%".format(progress))

        rmse = np.sqrt(np.mean(predictions))
        print("this fold rmse:{:.2f}".format(rmse))

        return rmse



class IterationEstimator(Estimator):
    """适合迭代式算法"""

    def train(self, train_dataset):
        self._prepare(train_dataset)
        for current_epoch in range(self.n_epochs):
            print(" processing epoch {}".format(current_epoch))
            self._iteration()
            print(" cur train rmse {}".format(self._rmse()))

    #准备工作
    def _prepare(self, train_dataset):
        raise NotImplementedError()

    #核心迭代
    def _iteration(self):
        raise NotImplementedError()

    #整体预测
    def _pred(self):
        raise NotImplementedError()

    #整体rmse
    def _rmse(self):
        pred_ratings = self._pred()
        real_ratings = self.train_dataset.matrix
        idx = real_ratings.nonzero()
        bias = np.asarray(pred_ratings[idx] - real_ratings[idx])
        return np.sqrt(np.sum(bias ** 2) / real_ratings.count_nonzero())