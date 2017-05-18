from __future__ import division, print_function

import numpy as np

class Estimator(object):

    def __init__(self):
        pass

    def train(self, train_dataset):
        raise NotImplementedError()

    def predict(self, u, i, r):
        raise NotImplementedError()

    def estimate(self, raw_test_dataset):
        l = len(raw_test_dataset)
        predictions = []
        m = 0
        for raw_u, raw_i, r, _ in raw_test_dataset:
            m += 1

            if not raw_u in self.train_dataset.uid_dict or \
                    not raw_i in self.train_dataset.iid_dict:
                real, est = r, self.train_dataset.global_mean
            else:
                u = self.train_dataset.uid_dict[raw_u]
                i = self.train_dataset.iid_dict[raw_i]
                real, est = self.predict(u, i, r)

            est = min(5, est)
            est = max(1, est)
            predictions.append((real - est) ** 2)
            if m%300 == 0:
                progress = 100 * (m / l)
                print("progress: %.2f%%" % progress)
        return np.sqrt(np.mean(predictions))