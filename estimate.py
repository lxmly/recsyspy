
import numpy as np

class Estimator(object):

    def __init__(self):
        pass

    def predict(self, u, i, r):
        raise NotImplementedError()

    def estimate(self, test_dataset):
        print("total {} ratings".format(test_dataset.matrix.nnz))
        predictions = []
        m = 0
        for u, i, r in test_dataset.all_ratings():
            m += 1
            real, est = self.predict(u, i, r)

            est = min(5, est)
            est = max(1, est)

            predictions.append((real - est) ** 2)
            if m%100 == 0:
                print("current {}th".format(m))
        return np.sqrt(np.mean(predictions))