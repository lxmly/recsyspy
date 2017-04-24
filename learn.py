from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix

import numpy as np
from scipy.sparse import random

if __name__ == '__main__':
    row = np.array([0, 3, 1, 0])
    col = np.array([0, 3, 1, 2])
    data = np.array([4, 5, 7, 9])

    matrix = csc_matrix((data, (row, col)))
    print(matrix.getrow(0).toarray())
    print(matrix.getrow(0).tocoo().nnz)
    print(matrix.getrow(0).tocoo().data)
