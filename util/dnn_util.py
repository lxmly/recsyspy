from keras.callbacks import Callback
from sklearn.metrics import mean_squared_error
import numpy as np


class RMSEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0, batch_size=2000)
            score = np.sqrt(mean_squared_error(self.y_val, y_pred))
            print("\n RMSE - epoch: %d - score: %.6f \n" % (epoch + 1, score))