import os

from algorithm.mf.baseline import Baseline
from util.databuilder import DataBuilder
from algorithm.mf.als import ExplicitALS
from algorithm.mf.svd import SVD
from algorithm.mf.svdpp import SVDPlusPlus
from algorithm.neighbor.slop_one import SlopOne
from algorithm.neighbor.itemcf import Itemcf


file_name = os.path.abspath("data/u.data")
data_builder = DataBuilder(file_name, k_folds=5, just_test_one=True)


def test_itemcf():
    data_builder.rmse(Itemcf())


def test_slopOne():
    data_builder.rmse(SlopOne())


def test_baseline():
    data_builder.rmse(Baseline())


def test_svd():
    data_builder.rmse(SVD())


def test_svdpp():
    data_builder.rmse(SVDPlusPlus())


def test_als():
    data_builder.rmse(ExplicitALS())