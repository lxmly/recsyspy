import pytest

from util.databuilder import DataBuilder
from algorithm.itemcf import Itemcf
from algorithm.slop_one import SlopOne
import os

file_name = os.path.abspath("data/u.data")
data_builder = DataBuilder(file_name, k_folds=7)

def test_itemcf():
    data_builder.rmse(Itemcf())

def test_slopOne():
    data_builder.rmse(SlopOne())



