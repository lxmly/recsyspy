import scipy.sparse as sparse
from scipy.sparse import lil_matrix
from slop_one import SlopOne
import numpy as np

from itemCF import ItemCF
from scipy.sparse import random
from matrix import Matrix
from scipy.stats import cosine

if __name__ == '__main__':
    row = np.array([0, 0, 1, 1, 2, 2])
    col = np.array([1, 2, 1, 2, 1, 2])
    data = np.array([2, 1, 2, 1, 2, 1])

    matrix = sparse.csc_matrix((data, (row, col))).todok()
    print matrix.A

    print sparse.eye(3)



    # matrix.data[1] = 100
    #
    # print(matrix.A)

    # itemCF = ItemCF()
    # itemCF.train(Matrix(matrix))
    #
    # print(itemCF.sim.A)
    #
    # print([1,2,3,4,5][0:30])


    # matrix[matrix < 3] = 0
    # print(matrix.A)

