# -*- coding:utf-8 -*-

import numpy as np

class SVDpp:
    def __init__(self, n_factors=20, n_epochs=20, lr=0.007, reg=.002):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
    def train(self, coo_matrix):
        user_num = coo_matrix.shape[0]
        item_num = coo_matrix.shape[1]

        #global mean
        global_mean = coo_matrix.sum() / coo_matrix.nnz

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
            for u, i, r in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
                k += 1
                if k % 100 == 0:
                    print(" processing line {}".format(k))

                #用户u点评的item数
                Nu = coo_matrix.getrow(u).tocoo().col
                I_Nu = coo_matrix.getrow(u).nnz
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
                    y[j] += self.lr * (e_ui  * q[j] / sqrt_N_u - self.reg * y[j])




