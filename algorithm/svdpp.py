# -*- coding:utf-8 -*-
from __future__ import division, print_function

import numpy as np
from estimator import Estimator


class SVDPlusPlus(Estimator):

    def __init__(self, n_factors=20, n_epochs=20, lr=0.007, reg=.002):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg

    def train(self, train_dataset):
        user_num = train_dataset.matrix.shape[0]
        item_num = train_dataset.matrix.shape[1]
        self.train_dataset = train_dataset

        #global mean
        global_mean = train_dataset.global_mean

        #user bias
        bu = np.zeros(user_num, np.double)

        #item bias
        bi = np.zeros(item_num, np.double)

        #user factor
        p = np.zeros((user_num, self.n_factors), np.double) + .1

        #item factor
        q = np.zeros((item_num, self.n_factors), np.double) + .1

        #item preference facotor
        y = np.zeros((item_num, self.n_factors), np.double) + .1

        for current_epoch in range(self.n_epochs):
            print(" processing epoch {}".format(current_epoch))
            k=0
            for u, i, r in train_dataset.all_ratings():
                k += 1
                if k % 100 == 0:
                    print(" processing line {}".format(k))

                #用户u点评的item集
                Nu = train_dataset.get_user(u)[0]
                I_Nu = len(Nu)
                sqrt_N_u = np.sqrt(I_Nu)

                #基于用户u点评的item集推测u的implicit偏好
                y_u = np.sum(y[Nu], axis=0)

                u_impl_prf = y_u / sqrt_N_u

                #预测值
                rp = global_mean + bu[u] + bi[i] + np.dot(q[i], p[u] + u_impl_prf)

                #误差
                e_ui = r - rp

                #sgd
                bu[u] += self.lr * (e_ui - self.reg * bu[u])
                bi[u] += self.lr * (e_ui - self.reg * bi[u])
                p[u] += self.lr * (e_ui * q[i] - self.reg * p[u])
                q[i] += self.lr * (e_ui * (p[u] + u_impl_prf) - self.reg * q[i])
                for j in Nu:
                    y[j] += self.lr * (e_ui * q[j] / sqrt_N_u - self.reg * y[j])

        self.global_mean = global_mean
        self.bu = bu
        self.bi = bi
        self.q = q
        self.p = p
        self.y = y

    def predict(self, u, i, r):
        Nu = self.train_dataset.get_user(u)[0]
        I_Nu = len(Nu)
        sqrt_N_u = np.sqrt(I_Nu)
        y_u = np.sum(self.y[Nu], axis=0) / sqrt_N_u

        est = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.q[i], self.p[u] + y_u)
        return r, est


