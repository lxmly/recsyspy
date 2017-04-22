from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix

import numpy as np


if __name__ == '__main__':
    row = np.array([0, 3, 1, 0])
    col = np.array([0, 3, 1, 2])
    data = np.array([4, 5, 7, 9])

    matrix = csc_matrix((data, (row, col)))
    print(matrix[:,:].toarray())
    print(matrix.transpose().nonzero())







