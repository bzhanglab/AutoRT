import numpy as np
from scipy.stats import iqr

def minMaxScale(x, min=0.0,max=120.0):
    new_x = 1.0*(x-min)/(max-min)
    return new_x

def minMaxScaleRev(x,min=0.0,max=120.0):
    old_x = x * (max - min) + min
    return old_x

def standardizeScale(x, mean=0.0, std=1.0):
    new_x = (x - mean) / std * 1.0
    return new_x

def standardizeScaleRev(x, mean=0.0, std=1.0):
    new_x = 1.0 * x * std + mean
    return new_x

def scaling_y(y, para=None):
    if "scaling_method" not in para.keys():
        # for old models.
        para['scaling_method'] = "min_max"
        para['rt_min'] = para['min_rt']
        para['rt_max'] = para['max_rt']

    scaling_method = para['scaling_method']
    y = np.asarray(y)
    if len(y.shape) >= 2:
        max_n = y.shape[0]
        if max_n < y.shape[1]:
            max_n = y.shape[1]
        y = y.reshape(max_n)
    if scaling_method == "min_max":
        #print("scaling method: min_max")
        new_y = minMaxScale(y,para['rt_min'],para['rt_max'])
    elif scaling_method == "mean_std":
        new_y = standardizeScale(y, para['rt_mean'], para['rt_std'])
    else:
        #print("scaling method: fixed scaling_factor")
        new_y = y/para['scaling_factor']

    return new_y

def scaling_y_rev(y, para=None):
    if "scaling_method" not in para.keys():
        # for old models.
        para['scaling_method'] = "min_max"
        para['rt_min'] = para['min_rt']
        para['rt_max'] = para['max_rt']

    scaling_method = para['scaling_method']
    y = np.asarray(y)
    if len(y.shape) >= 2:
        max_n = y.shape[0]
        if max_n < y.shape[1]:
            max_n = y.shape[1]
        y = y.reshape(max_n)
    if scaling_method == "min_max":
        #print("scaling method: min_max")
        new_y = minMaxScaleRev(y,para['rt_min'],para['rt_max'])
    elif scaling_method == "mean_std":
        new_y = standardizeScaleRev(y, para['rt_mean'], para['rt_std'])
    else:
        #print("scaling method: fixed scaling_factor")
        new_y = para['scaling_factor']*y
    return new_y


def combine_rts(x, reverse=False, method="mean", remove_outlier=True, scale_para=None):
    n_1 = len(x)
    if reverse == True:
        #print("Reverse: %s" % str(reverse))
        x = scaling_y_rev(x,scale_para)

    if remove_outlier == True:
        #print("Remove outlier: %s" % str(remove_outlier))
        r1 = np.percentile(x, 25) - 1.5 * iqr(x)
        r2 = np.percentile(x, 75) + 1.5 * iqr(x)
        x = x[(x >= r1) & (x <= r2)]
        n_2 = len(x)
        #print("remove %d" % (n_1 - n_2))
    if method == "mean":
        #print("Combine method: %s" % str(method))
        res = np.mean(x)
    else:
        #print("Combine method: %s" % str(method))
        res = np.median(x)
        #res = np.percentile(x, 50)

    return (res)