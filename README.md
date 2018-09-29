# Recsyspy
Classic recommendation algorithms implementation

## Algorithm
|DNN Model |RMSE|MAE
| :-------- |:--------|:-------- |
|NeuMF|0.9433|0.7485   

|MF Model | RMSE     | MAE
| :-------- | :-------- | :-------- |
| Baseline  | 0.946|0.742 
| SVD|0.931|0.731|
| SVDPlusPlus|0.927|0.726
| Explicit ALS  |1.199|0.903
| Implicit ALS |2.752|2.525

|Neighborhood Model |RMSE|MAE
| :-------- |:--------|:-------- |
|Itemcf|1.029|0.802
|WeightedSlopOne|1.043|0.835|

## Example
```python
import os

from util.databuilder import DataBuilder
from algorithm.dnn.neumf import NeuMF

file_name = os.path.abspath("data/ml-100k/u.data")
data_builder = DataBuilder(file_name, just_test_one=True)


data_builder.eval(NeuMF(epochs=2), k_folds=5)
```


## Dateset
* MovieLens 

## Papers
### Dnn Algorithm
* Neural Collaborative Filtering

### MF Algorithm  
* Yehuda Koren. Factorization meets the neighborhood: a multifaceted collaborative filtering model
* Matrix factorization techniques for recommender systems
* Advances in Collaborative Filtering

### Neighborhood Algorithm
* Slope one predictors for online rating-based collaborative filtering


