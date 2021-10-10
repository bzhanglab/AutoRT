
from numpy.random import seed
seed(2019)
#from tensorflow import set_random_seed
#set_random_seed(2020)
from sklearn.model_selection import train_test_split
import sys


import matplotlib
matplotlib.use("agg")
import os
import json

from .PeptideEncode import *
from .Utils import scaling_y, scaling_y_rev


def data_processing(input_data: str, test_file=None, mod=None, max_x_length = 50, scale_para=None, unit="s",
                    out_dir="./", aa_file=None, add_reverse=False, random_seed=2018, seq_encode_method="one_hot"):
    """
    Used by AutoRT
    Processing training data
    :param input_data:
    :param test_file:
    :param mod:
    :param max_x_length:
    :param min_rt:
    :param max_rt:
    :param unit:
    :param out_dir:
    :param aa_file:
    :return:
    """

    if aa_file is not None:
        ## read aa information from file
        load_aa(aa_file)
    else:
        if mod is not None:
            add_mod(mod)

        aa2file = out_dir + "/aa.tsv"
        save_aa(aa2file)

    ##
    siteData = pd.read_table(input_data, sep="\t", header=0, low_memory=False)

    if "x" not in siteData.columns:
        siteData.columns = ['x','y']

    if unit.startswith("s"):
        siteData['y'] = siteData['y']/60.0

    if "rt_max" in scale_para.keys():
        max_rt = scale_para['rt_max']
    else:
        max_rt = 0

    if max_rt < siteData['y'].max():
        max_rt = siteData['y'].max() + 1.0

    if "rt_min" in scale_para.keys():
        min_rt = scale_para['rt_min']
    else:
        min_rt = 0.0

    if min_rt > siteData['y'].min():
        min_rt = siteData['y'].min() - 1.0

    # aaMap = getAAcodingMap()
    n_aa_types = len(letterDict)
    print("AA types: %d" % (n_aa_types))


    ## all aa in data
    all_aa = set()

    ## get the max length of input sequences

    longest_pep_training_data = 0
    for pep in siteData["x"]:
        if max_x_length < len(pep):
            max_x_length = len(pep)

        if longest_pep_training_data < len(pep):
            longest_pep_training_data = len(pep)

        ##
        for aa in pep:
            all_aa.add(aa)



    print("Longest peptide in training data: %d\n" % (longest_pep_training_data))

    ## test data
    test_data = None
    longest_pep_test_data = 0
    if test_file is not None:
        print("Use test file %s" % (test_file))
        test_data = pd.read_table(test_file, sep="\t", header=0, low_memory=False)
        if "x" not in test_data.columns:
            test_data.columns = ['x', 'y']
        if unit.startswith("s"):
            test_data['y'] = test_data['y'] / 60.0

        if max_rt < test_data['y'].max():
            max_rt = test_data['y'].max() + 1.0

        if min_rt > test_data['y'].min():
            min_rt = test_data['y'].min() - 1.0

        for pep in test_data["x"]:
            if max_x_length < len(pep):
                max_x_length = len(pep)

            if longest_pep_test_data < len(pep):
                longest_pep_test_data = len(pep)

            for aa in pep:
                all_aa.add(aa)

        print("Longest peptide in test data: %d\n" % (longest_pep_test_data))

    print(sorted(all_aa))

    siteData = siteData.sample(siteData.shape[0], replace=False, random_state=2018)

    #if add_reverse is True:
    #    train_data = np.zeros((siteData.shape[0], 2*max_x_length, n_aa_types))
    #else:
    #    train_data = np.zeros((siteData.shape[0], max_x_length, n_aa_types))

    train_data = encodePeptides(siteData,max_length=max_x_length,seq_encode_method=seq_encode_method)
    if seq_encode_method == "one_hot":
        train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[2]))

    X_test = np.empty(1)
    Y_test = np.empty(1)

    print("RT range: %d - %d\n" % (min_rt,max_rt))


    if scale_para['scaling_method'] == "mean_std":
        scale_para['rt_mean'] = float(np.mean(siteData['y']))
        scale_para['rt_std'] = float(np.std(siteData['y']))
        print("RT mean: %d, std: %d" % (scale_para['rt_mean'], scale_para['rt_std']))


    scale_para['rt_max'] = max_rt
    scale_para['rt_min'] = min_rt

    if test_data is None:
        X_train, X_test, Y_train, Y_test = train_test_split(train_data,
                                                            #to_categorical(pos_neg_all_data['y'], num_classes=2),
                                                            scaling_y(siteData['y'], scale_para),
                                                            test_size=0.1, random_state=100)
    else:
        X_train = train_data
        #Y_train = to_categorical(pos_neg_all_data['y'], num_classes=2)
        Y_train = siteData['y']
        Y_train = scaling_y(Y_train, scale_para)
        if len(Y_train.shape) >= 2:
            Y_train = Y_train.reshape(Y_train.shape[1])

        if add_reverse is True:
            X_test = np.zeros((test_data.shape[0], 2*max_x_length, n_aa_types))
        else:
            X_test = np.zeros((test_data.shape[0], max_x_length, n_aa_types))

        X_test = encodePeptides(test_data,max_length=max_x_length,seq_encode_method=seq_encode_method)
        if seq_encode_method == "one_hot":
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        Y_test = scaling_y(test_data['y'],scale_para)

    if seq_encode_method == "one_hot":
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

    print("X_train shape:")
    print(X_train.shape)
    print("X_test shape:")
    print(X_test.shape)
    print("Modeling start ...")


    #print("save training data to file:\n")
    #np.save("X_train.npy", X_train)
    #np.save("y_train.npy", Y_train)
    #np.save("X_test.npy", X_test)
    #np.save("y_test.npy", Y_test)

    return [X_train, Y_train, X_test, Y_test, scale_para]


def get_max_length_from_input_data(input_data:str):
    siteData = pd.read_table(input_data, sep="\t", header=0, low_memory=False)
    longest_pep = 0
    for pep in siteData["x"]:
        if longest_pep < len(pep):
            longest_pep = len(pep)

    return longest_pep

def processing_prediction_data(model_file: str, input_data: str, seq_encode_method="one_hot"):
    '''
    Used by AutoRT
    :param model_file: model file in json format
    :param input_data: prediction file
    :return: A numpy matrix for prediction
    '''

    with open(model_file, "r") as read_file:
        model_list = json.load(read_file)

    model_folder = os.path.dirname(model_file)
    aa_file = model_folder + "/" + os.path.basename(model_list['aa'])
    load_aa(aa_file)

    ##
    siteData = pd.read_table(input_data, sep="\t", header=0, low_memory=False)

    n_aa_types = len(letterDict)
    print("AA types: %d" % (n_aa_types))

    ## all aa in data
    all_aa = set()

    ## get the max length of input sequences
    max_x_length = int(model_list['max_x_length'])

    if "add_reverse" in model_list.keys():
        if model_list['add_reverse'] == 1:
            add_reverse = True
        else:
            add_reverse = False
    else:
        add_reverse = False

    longest_pep_len = 0
    for pep in siteData["x"]:
        #if max_x_length < len(pep):
        #    max_x_length = len(pep)
        if longest_pep_len < len(pep):
            longest_pep_len = len(pep)
        ##
        for aa in pep:
            all_aa.add(aa)

    print("Longest peptide in input data: %d\n" % (longest_pep_len))

    print(sorted(all_aa))

    # siteData = siteData.sample(siteData.shape[0], replace=False, random_state=2018)
    #if add_reverse is True:
    #    train_data = np.zeros((siteData.shape[0], 2*max_x_length, n_aa_types))
    #else:
    #    train_data = np.zeros((siteData.shape[0], max_x_length, n_aa_types))

    #k = 0
    #for i, row in siteData.iterrows():
    #    peptide = row['x']
    #    train_data[k] = encodePeptideOneHot(peptide, max_length=max_x_length, add_reverse=add_reverse)
    #    k = k + 1

    train_data = encodePeptides(siteData,max_length=max_x_length,seq_encode_method=seq_encode_method)
    if seq_encode_method == "one_hot":
        train_data = train_data.reshape((train_data.shape[0], train_data.shape[1], train_data.shape[2]))
        train_data = train_data.astype('float32')


    return train_data


def split_data_file(file,test_size=0.1,random_state=2020,out_dir="./"):
    df = pd.read_table(file,sep="\t",header=0)
    train, test = train_test_split(df,test_size=test_size,random_state=random_state, shuffle=True)
    train_file = out_dir + "/train.tsv"
    test_file = out_dir + "/validation.tsv"
    train.to_csv(train_file, index=False, sep="\t")
    test.to_csv(test_file, index=False, sep="\t")
    return [train_file,test_file]









