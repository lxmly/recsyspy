import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve

from algorithm.estimator import Estimator
from util.matrix import Matrix


class ExplicitALS(Estimator):

    def __init__(self, n_factors=20, n_epochs=20, reg=.1):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg

    def als(self, X, Y, is_user):
        print 'num %d' % (X.shape[0])
        YTY_regI = Y.T.dot(Y) + self.reg * sparse.eye(self.n_factors)
        uuids = self.train_dataset.uids if is_user else self.train_dataset.iids

        m = 0
        for u in uuids:
            if m % 100 == 0:
                print 'cur %d th u' % (m)
            m += 1
            ru = self.train_dataset.matrix[u,:].T.A if is_user else self.train_dataset.matrix[:,u].A
            X[u] = spsolve(YTY_regI, Y.T.dot(ru))
        return X

    def normaliseRow(self, x):
        return x / sum(x)

    def initialiseMatrix(self, n, f):
        return np.apply_along_axis(self.normaliseRow, 1, abs(np.random.randn(n, f)))

    def train(self, train_dataset):
        user_num = train_dataset.matrix.shape[0]
        item_num = train_dataset.matrix.shape[1]
        self.train_dataset = train_dataset

        X = sparse.csr_matrix(self.initialiseMatrix(user_num, self.n_factors))
        Y = sparse.csr_matrix(self.initialiseMatrix(item_num, self.n_factors))

        for k in range(self.n_epochs):
            print("the {}th epochs.".format(k))
            X = self.als(X, Y, True)
            Y = self.als(Y, X, False)
        self.X = Matrix(X)
        self.Y = Matrix(Y)

    def predict(self, u, i, r):
        est = self.X.matrix[u, :].dot(self.Y.matrix[i, :].T)[0, 0]
        return r, est





