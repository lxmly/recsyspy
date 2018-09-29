import os

from algorithm.mf.baseline import Baseline
from util.databuilder import DataBuilder

from algorithm.dnn.neumf import NeuMF

from algorithm.mf.explicit_als import ExplicitALS
from algorithm.mf.svd import SVD
from algorithm.mf.svdpp import SVDPlusPlus
from algorithm.mf.implicit_als import ImplicitALS

from algorithm.neighborhood.slop_one import SlopOne
from algorithm.neighborhood.itemcf import Itemcf

file_name = os.path.abspath("data/ml-100k/u.data")
data_builder = DataBuilder(file_name, k_folds=5, just_test_one=True)


def test_neumf():
    data_builder.eval(NeuMF(epochs=2))


def test_itemcf():
    data_builder.eval(Itemcf())


def test_slopOne():
    data_builder.eval(SlopOne())


def test_baseline():
    data_builder.eval(Baseline())


def test_svd():
    data_builder.eval(SVD())


def test_svdpp():
    data_builder.eval(SVDPlusPlus())


def test_explicit_als():
    data_builder.eval(ExplicitALS())


def test_implicit_als():
    data_builder.eval(ImplicitALS())