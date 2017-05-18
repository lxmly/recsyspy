import pytest

from util.databuilder import DataBuilder
from algorithm.itemcf import Itemcf

file_name = '/Users/fanruiqiang/work/data/ml-100k/u.data'
data_builder = DataBuilder(file_name, k_folds=7)

def test_itemcf():
    itemcf = Itemcf()
    data_builder.rmse(itemcf)



