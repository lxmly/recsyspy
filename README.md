# recsyspy
经典推荐算法实现
* 基于scipy 稀疏矩阵构建数据模型
* 使用k折交叉验证测试算法
* 抽象算法过程，易于扩展

## 算法
| 矩阵分解模型 | RMSE     | 
| :-------- | :-------- |
| Baseline  | 0.946| 
| SVD|0.931|
| SVDPlusPlus|0.927|
| Explicit ALS  |1.199|
| Implicit ALS |2.752|

|邻居模型 |RMSE|
| :-------- |:--------|
|Itemcf|1.029|
|WeightedSlopOne|1.043|


## 数据集
* MovieLens 

## 依赖
* scipy
* numpy

## 参考paper
* Yehuda Koren. Factorization meets the neighborhood: a multifaceted collaborative filtering model
* Matrix factorization techniques for recommender systems
* Advances in Collaborative Filtering
* Slope one predictors for online rating-based collaborative filtering