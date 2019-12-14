
import sklearn
from sklearn.model_selection import train_test_split
from autort.Utils import scaling_y_rev
import pandas as pd
import numpy as np
import scipy
from scipy.stats import pearsonr



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
