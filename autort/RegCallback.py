
import scipy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import Callback
from .Metrics import calc_delta_t95
from .Utils import scaling_y_rev
import sklearn
from scipy.stats import pearsonr
import pandas as pd
import numpy as np


class RegCallback(Callback):
    """
    Used by AutoRT
    """

    def __init__(self, X_train, X_test, y_train, y_test, scale_para=None):
        self.x = X_train
        self.y = scaling_y_rev(y_train,scale_para) # minMaxScaleRev(y_train,min_rt,max_rt)
        self.x_val = X_test
        self.y_val = scaling_y_rev(y_test,scale_para) # minMaxScaleRev(y_test,min_rt,max_rt)
        self.scale_para = scale_para
        print(scale_para)

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs=None):

        ## training data
        y_pred = self.model.predict(self.x)
        y_pred_rev = scaling_y_rev(y_pred, self.scale_para)

        y1 = pd.DataFrame({"y": self.y, "y_pred": y_pred_rev.reshape(y_pred_rev.shape[0])})
        cor1 = pearsonr(y1['y'],y1['y_pred'])[0]
        mae1 = sklearn.metrics.mean_absolute_error(y1['y'],y1['y_pred'])
        r21 = sklearn.metrics.r2_score(y1['y'],y1['y_pred'])
        abs_median1 = np.median(np.abs(y1['y'] - y1['y_pred']))
        d_t951 = calc_delta_t95(y1['y'], y1['y_pred'])
        ## test data
        y_pred_val = self.model.predict(self.x_val)
        y_pred_val_rev = scaling_y_rev(y_pred_val, self.scale_para)
        y2 = pd.DataFrame({"y": self.y_val, "y_pred": y_pred_val_rev.reshape(y_pred_val_rev.shape[0])})
        cor2 = pearsonr(y2['y'], y2['y_pred'])[0]
        mae2 = sklearn.metrics.mean_absolute_error(y2['y'], y2['y_pred'])
        r22 = sklearn.metrics.r2_score(y2['y'], y2['y_pred'])
        abs_median2 = np.median(np.abs(y2['y'] - y2['y_pred']))
        d_t952 = calc_delta_t95(y2['y'], y2['y_pred'])
        print('\nCor: %s - Cor_val: %s, MAE: %s - MAE_val: %s, R2: %s - R2_val: %s, MedianE: %s - MedianE_val: %s, dt95: %s - dt95_val: %s' % (str(round(cor1, 4)), str(round(cor2, 4)),
                                                                                       str(round(mae1, 4)), str(round(mae2, 4)),
                                                                                       str(round(r21, 4)), str(round(r22, 4)),
                                                                                       str(round(abs_median1, 4)), str(round(abs_median2, 4)),
                                                                                       str(round(d_t951, 4)), str(round(d_t952, 4))), end=100 * ' ' + '\n')

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
