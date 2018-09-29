# -*- coding:utf-8 -*-
"""NeuMF model
Paper: Neural Collaborative Filtering
Apply dnn on MF
"""

from __future__ import division, print_function

import numpy as np
from algorithm.estimator import Estimator
from keras.layers import Input, Embedding, Dense, Flatten,\
                         BatchNormalization, Dropout
from keras.layers import multiply, concatenate
from keras.models import Model

class NeuMF(Estimator):
    """
    mf_dim: Integer.
       MF dimension.
    mlp_dim: Integer.
       MLP dimension
    epochs: Integer 
       Number of epochs to train the model
    """

    def __init__(self, mf_dim=12, mlp_dim=12, epochs=2):
        self.mf_dim = mf_dim
        self.mlp_dim = mlp_dim
        self.epochs = epochs

    def transform(self, dateset):
        X = {}
        u, i, r = dateset.all_ratings(axis=0)
        X['user_idx'] = u.reshape(-1, 1)
        X['item_idx'] = i.reshape(-1, 1)

        y = r.reshape(-1, 1)

        return X, y

    def get_neumf_model(self, user_num, item_num):
        user_input = Input(shape=[1], name="user_idx")
        item_input = Input(shape=[1], name="item_idx")

        mf_embedding_user = Embedding(user_num, self.mf_dim)(user_input)
        mf_embedding_item = Embedding(item_num, self.mf_dim)(item_input)

        gmf_layer = multiply([mf_embedding_user, mf_embedding_item])

        mlp_embedding_user = Embedding(user_num, self.mlp_dim)(user_input)
        mlp_embedding_item = Embedding(item_num, self.mlp_dim)(item_input)

        mlp_layer = concatenate([mlp_embedding_user, mlp_embedding_item])

        mlp_layer = BatchNormalization()(mlp_layer)
        mlp_layer = Dense(32)(mlp_layer)
        mlp_layer = Dense(16)(mlp_layer)
        mlp_layer = Dense(8)(mlp_layer)
        mlp_layer = Dense(4)(mlp_layer)
        mlp_layer = Dropout(0.5)(mlp_layer)

        neumf_layer = concatenate([gmf_layer, mlp_layer])
        neumf_layer = Flatten()(neumf_layer)
        pred = Dense(1)(neumf_layer)

        model = Model(inputs=[user_input, item_input], outputs=pred)
        model.compile(optimizer='adam', loss='mse')

        return model


    def _train(self):
        user_num = self.train_dataset.matrix.shape[0]
        item_num = self.train_dataset.matrix.shape[1]
        X_train, y_train = self.transform(self.train_dataset)

        self.neumf_model = self.get_neumf_model(user_num, item_num)
        self.neumf_model.fit(X_train, y_train, epochs=self.epochs)

    def predict(self, u, i):
        #not batch but single pred
        X = {}
        X['user_idx'] = np.array([u])
        X['item_idx'] = np.array([i])

        return self.neumf_model.predict(X)[0, 0]