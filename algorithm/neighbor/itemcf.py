# -*- coding:utf-8 -*-
from __future__ import division, print_function

import numpy as np
from scipy.sparse import lil_matrix

from algorithm.mf.estimator import Estimator


class Itemcf(Estimator):

    """
    Attributes
    ---------
    min : tuple
       coo矩阵的上线限制 
    """

    def __init__(self, min=2, topk=20):
        self.min = min
        self.topk = topk

    def compute_cosine_similarity(self, user_num, item_num, users_ratings):
        sim = lil_matrix((item_num, item_num), dtype=np.double)

        #点积
        dot = lil_matrix((item_num, item_num), dtype=np.double)

        #左向量平方和
        sql = lil_matrix((item_num, item_num), dtype=np.double)

        #右向量平方和
        sqr = lil_matrix((item_num, item_num), dtype=np.double)

        #共现矩阵
        coo = lil_matrix((item_num, item_num), dtype=np.double)

        m = 1
        for u, (ii, rr) in users_ratings:
            m = m + 1
            for k in range(len(ii) - 1):
                k1, k2 = k, k+1
                i1, i2 = ii[k1], ii[k2]
                if i1 > i2:
                    i1, i2 = i2, i1
                    k1, k2 = k2, k1
                dot[i1, i2] += rr[k1] * rr[k2]
                sql[i1, i2] += rr[k1]**2
                sqr[i1, i2] += rr[k2]**2
                coo[i1, i2] += 1

            if m % 50 == 0:
                progress = 100 * (m / user_num)
                print("coo progress: %.2f%%" % progress)



        #dok_matrix不适合进行矩阵算术操作，转为csc格式
        dot = dot.tocsc()
        sql = sql.tocsc()
        sqr = sqr.tocsc()
        coo = coo.tocsc()

        #交互数低于限制全部清零
        dot.data[coo.data < self.min] = 0

        #左右向量平方和的乘积
        sql.data *= sqr.data

        #只需要考虑非0点积
        row, col = dot.nonzero()

        #cosine相似矩阵
        sim[row, col] = dot[row, col] / np.sqrt((sql)[row, col])
        sim[col, row] = sim[row, col]

        return sim

    def train(self, train_dataset):
        print("total {} user, {} ratings".format(train_dataset.matrix.shape[0], train_dataset.matrix.nnz))
        user_num = train_dataset.matrix.shape[0]
        item_num = train_dataset.matrix.shape[1]
        self.sim = self.compute_cosine_similarity(user_num, item_num, train_dataset.get_users())
        self.item_means = train_dataset.get_item_means()
        self.user_means = train_dataset.get_user_means()
        self.train_dataset = train_dataset
        print("train end")

    def predict(self, u, i, r):
        ll, rr = self.train_dataset.get_user(u)
        neighbors = [(sim_i, self.sim[i, sim_i], sim_r) for sim_i, sim_r in zip(ll, rr)]

        neighbors = sorted(neighbors, key=lambda tple: tple[1], reverse=True)[0:self.topk]
        est = self.item_means[i]
        sum = 0
        divisor = 0

        for sim_i, sim, sim_r in neighbors:
            sum += sim * (sim_r - self.item_means[sim_i])
            divisor += sim

        if divisor != 0:
            est += sum / divisor
        return r, est