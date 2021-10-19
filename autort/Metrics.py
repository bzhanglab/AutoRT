
import sklearn
from sklearn.model_selection import train_test_split
from autort.Utils import scaling_y_rev
import pandas as pd
import numpy as np
import scipy
from scipy.stats import pearsonr
from itertools import combinations
import multiprocessing
from .Utils import combine_rts
import sys
import psutil
import functools





def evaluate_model(y_t, y_p, para=None, out_dir="./", prefix="test", reverse=True):
    if reverse == True:
        #y_t = minMaxScaleRev(y_t, para['min_rt'], para['max_rt'])
        #y_p = minMaxScaleRev(y_p, para['min_rt'], para['max_rt'])
        y_t = scaling_y_rev(y_t, para)
        y_p = scaling_y_rev(y_p, para)
    y2 = pd.DataFrame({"y": y_t, "y_pred": y_p.reshape(y_p.shape[0])})
    cor = pearsonr(y2['y'], y2['y_pred'])[0]
    mae = sklearn.metrics.mean_absolute_error(y2['y'], y2['y_pred'])
    r2 = sklearn.metrics.r2_score(y2['y'], y2['y_pred'])
    abs_median = float(np.median(np.abs(y2['y'] - y2['y_pred'])))
    d_t95 = calc_delta_t95(y2['y'], y2['y_pred'])
    print('Cor: %s, MAE: %s, R2: %s, abs_median_e: %s, dt95: %s' % (
        str(round(cor, 4)), str(round(mae, 4)),
        str(round(r2, 4)), str(round(abs_median, 4)),
        str(round(d_t95, 4))), end=100 * ' ' + '\n')

    ## output
    out_file = out_dir + "/" + prefix + ".csv"
    y2.to_csv(out_file)



def calc_delta_t95(obs, pred):
    q95 = int(np.ceil(len(obs) * 0.95))
    return 2 * sorted(abs(obs - pred))[q95 - 1]

def calc_elta_tr95(obs, pred):
    return calc_delta_t95(obs, pred) / (max(obs) - min(obs))


class ModelE:

    def __init__(self, x, y, para, reverse=True, metric="median_absolute_error"):
        self.x = x
        self.y = y
        self.para = para
        self.reverse = reverse
        self.metric = metric

    def evaluate_model_combn(self, i):
        ind = list()
        ind.extend(i)
        y_p = np.apply_along_axis(combine_rts, 1, self.x[ind], reverse=self.reverse, scale_para=self.para, method="mean",remove_outlier=True)
        y_t = self.y
        if self.reverse == True:
            y_t = scaling_y_rev(y_t, self.para)
            #y_p = scaling_y_rev(y_p, para)
        y2 = pd.DataFrame({"y": y_t, "y_pred": y_p.reshape(y_p.shape[0])})
        #if metric == "median_absolute_error":

        cor = pearsonr(y2['y'], y2['y_pred'])[0]
        mean_absolute_error   = sklearn.metrics.mean_absolute_error(y2['y'], y2['y_pred'])
        # median_absolute_error = float(np.median(np.abs(y2['y'] - y2['y_pred'])))
        median_absolute_error = sklearn.metrics.median_absolute_error(y2['y'], y2['y_pred'])
        r2 = sklearn.metrics.r2_score(y2['y'], y2['y_pred'])
        d_t95 = calc_delta_t95(y2['y'], y2['y_pred'])
        return [i,cor,r2,mean_absolute_error,median_absolute_error,d_t95]


def model_selection(x, y, para=None, metric="median_absolute_error"):
    ## x is a numpy array. shape: sample vs model
    n_models = x.shape[1]
    print("# of models: %d " % (n_models))
    models_list = list()
    for i in range(n_models):
        models_list.extend(list(combinations(range(n_models),i+1)))

    if sys.platform.lower().startswith("win"):
        print("Run on Windows system:")
        p = multiprocessing.Pool(psutil.cpu_count(logical=False))
    else:
        print("Run on %s system:" % sys.platform)
        p = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    model_e = ModelE(x=x, y=y, para=para, reverse=True, metric=metric)
    # evaluate_model_combn_x = functools.partial(evaluate_model_combn, x=x,y=y,para=para,reverse=True,metric=metric)
    data = p.map(model_e.evaluate_model_combn, models_list)
    p.close()
    ## select best combination
    best_i = None
    best_metric = None
    if metric == "median_absolute_error":
        mae = np.Inf
        for i in data:
            if mae > i[4]:
                best_i = i
                mae = i[4]
        best_metric = mae
    elif metric == "r2":
        r2 = -np.Inf
        for i in data:
            if r2 < i[2]:
                best_i = i
                r2 = i[2]
        best_metric = r2

    print("Best model combination based on metric: %s" % (metric))
    print(best_i)
    return [best_i,best_metric,data]

