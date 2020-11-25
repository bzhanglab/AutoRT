
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

def evaluate_model_combn(i, x, y, para=None, reverse=True, metric="median_absolute_error"):
    ind = list()
    ind.extend(i)
    y_p = np.apply_along_axis(combine_rts, 1, x[ind], reverse=reverse, scale_para=para, method="mean",remove_outlier=True)
    y_t = y
    if reverse == True:
        y_t = scaling_y_rev(y_t, para)
        #y_p = scaling_y_rev(y_p, para)
    y2 = pd.DataFrame({"y": y_t, "y_pred": y_p.reshape(y_p.shape[0])})
    metric_res = None
    if metric == "median_absolute_error":
        metric_res = float(np.median(np.abs(y2['y'] - y2['y_pred'])))
    return [i,metric_res]

def model_selection(x, y, para=None, metric="median_absolute_error"):
    ## x is a numpy array. shape: sample vs model
    n_models = x.shape[1]
    print("# of models: %d " % (n_models))
    models_list = list()
    for i in range(n_models):
        models_list.extend(list(combinations(range(n_models),i+1)))

    p = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    evaluate_model_combn_x = functools.partial(evaluate_model_combn, x=x,y=y,para=para,reverse=True,metric=metric)
    data = p.map(evaluate_model_combn_x, models_list)
    p.close()
    ## select best combination
    best_i = None
    if metric == "median_absolute_error":
        mae = np.Inf
        for i in data:
            if mae > i[1]:
                best_i = i
                mae = i[1]

    print("Best model combination based on metric: %s" % (metric))
    print(best_i)
    return [best_i,data]

