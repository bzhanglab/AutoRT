import math

from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, MaxPooling2D, Conv2D, Conv1D, Bidirectional, LSTM, \
    Embedding, MaxPooling1D, Average, GRU, LSTM, Bidirectional, GRU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model, load_model, clone_model, model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
import pandas as pd
import os
#from keras_radam import RAdam
#from keras_lookahead import Lookahead
import gc
import tensorflow.keras.backend as K
import tensorflow as tf

from .RegCallback import RegCallback
from .DataIO import data_processing, processing_prediction_data, get_max_length_from_input_data, split_data_file
from .Utils import scaling_y, scaling_y_rev, combine_rts
from .Metrics import evaluate_model,model_selection

#from .PolynomialDecay import PolynomialDecay
from multiprocessing import Pool
import math

import pickle
import json
import numpy as np
from shutil import copyfile
import sys
from .ModelT import ModelT, run_model_t
from .ModelUpdate import ModelUpdate
from multiprocessing import Process

#from .AdamW import AdamW




def build_default_model(input_shape):
    """
    Don't use this model.
    :param input_shape:
    :return:
    """
    model = Sequential()
    model.add(Conv1D(512, 3, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Conv1D(128, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Conv1D(64, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(2)))

    # model.add(Bidirectional(CuDNNGRU(50, return_sequences=True)))
    model.add(Bidirectional(GRU(50, return_sequences=True)))
    # model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    return model

def train_model(input_data: str, test_file=None, batch_size=64, nb_epoch=100, early_stop_patience=None, mod=None,
                max_x_length = 50, scale_para=None, unit="s",out_dir="./", prefix = "test",
                p_model=None,
                model=None, optimizer_name=None,use_radam=False,add_reverse=False,add_ReduceLROnPlateau=False,
                use_external_test_data=True,seq_encode_method="one_hot"):
    """
    Used by AutoRT
    :param input_data:
    :param test_file:
    :param batch_size:
    :param nb_epoch:
    :param early_stop_patience:
    :param mod:
    :param max_x_length:
    :param min_rt:
    :param max_rt:
    :param unit:
    :param out_dir:
    :param prefix:
    :param p_model:
    :param model:
    :param optimizer_name:
    :param use_radam:
    :param use_external_test_data:
    :return:
    """

    res_map = dict()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("Build deep learning model ...")

    X_train, Y_train, X_test, Y_test, scale_para = data_processing(input_data=input_data, test_file = test_file,
                                                                   mod = mod, max_x_length = max_x_length,
                                                                   scale_para=scale_para, unit = unit,out_dir=out_dir,
                                                                   add_reverse=add_reverse,
                                                                   seq_encode_method=seq_encode_method)

    print("Save training and testing data to file:\n")
    np.save("X_train", X_train)
    np.save("Y_train", Y_train)
    np.save("X_test", X_test)
    np.save("Y_test", Y_test)

    if model is None:
        print("Use default model ...")
        model = build_default_model(X_train.shape[1:])
    else:
        print("Use input model ...")
        model = clone_model(model)

    if p_model is not None:
        transfer_layer = 5
        frozen = True
        # model_copy.set_weights(model.get_weights())
        #if use_radam == True:
        #    base_model = load_model(p_model, custom_objects = {"Lookahead": Lookahead, "RAdam":RAdam})
        #else:
            #base_model = load_model(p_model)
        base_model = load_model(p_model)
        print("Perform transfer learning ...")
        n_layers = len(base_model.layers)
        print("The number of layers: %d" % (n_layers))
        for l in range((n_layers - transfer_layer)):
            if l != 0:
                model.layers[l].set_weights(base_model.layers[l].get_weights())
                if frozen is True:
                    model.layers[l].trainable = False
                    print("layer (frozen:True): %d %s" % (l,model.layers[l].name))
                else:
                    print("layer (frozen:False): %d %s" % (l,model.layers[l].name))

    if model.optimizer is None:
        ## use default optimizer: Adam
        if optimizer_name is None:

            #if use_radam == True:
            #    print("Use optimizer: %s" % ("rectified-adam"))
            #    model.compile(loss='mean_squared_error',
            #                  # math.ceil(X_train.shape[0]/batch_size*nb_epoch)
            #                  optimizer=Lookahead(RAdam(),sync_period=5, slow_step=0.5))
            #                  #optimizer=Lookahead(RAdam(total_steps=math.ceil(X_train.shape[0]/batch_size), warmup_proportion=0.1, min_lr=1e-5, lr=0.001),sync_period=5, slow_step=0.5),
            #                  #metrics=['mean_squared_error'])
            #else:
            #    print("Use default optimizer:Adam")
            #    model.compile(loss='mean_squared_error',
            #              optimizer="adam")
            #              #metrics=['mean_squared_error'])
            print("Use default optimizer:Adam")
            model.compile(loss='mean_squared_error',optimizer="adam")

        else:

            #if use_radam == True:
            #    print("Use optimizer: %s" % ("rectified-adam"))
            #    model.compile(loss='mean_squared_error',
            #                  optimizer=Lookahead(RAdam(),sync_period=5, slow_step=0.5))
            #                  #optimizer=Lookahead(RAdam(total_steps=math.ceil(X_train.shape[0]/batch_size), warmup_proportion=0.1, min_lr=1e-5, lr=0.001),
            #                  #                    sync_period=5, slow_step=0.5),
            #                  #metrics=['mean_squared_error'])
            #else:
            #    print("Use optimizer provided by user: %s" % (optimizer_name))
            #    model.compile(loss='mean_squared_error',
            #              optimizer=optimizer_name)
                          # Implementation from https://github.com/GLambard/AdamW_Keras
                          #optimizer=Adam(amsgrad=True))
                          #optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True))
                          #optimizer=Lookahead(AdamW(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0., weight_decay=1e-4, batch_size=batch_size, samples_per_epoch=X_train.shape[0], epochs=nb_epoch),sync_period=5, slow_step=0.5))
                          #metrics=['mean_squared_error'])
            print("Use optimizer provided by user: %s" % (optimizer_name))
            model.compile(loss='mean_squared_error',optimizer=optimizer_name)

    else:
        if optimizer_name is None:



            #if use_radam == True:
            #    print("Use optimizer: %s" % ("rectified-adam"))
            #    model.compile(loss='mean_squared_error',
            #                  optimizer=Lookahead(RAdam(),sync_period=5, slow_step=0.5))
            #                  #optimizer=Lookahead(RAdam(total_steps=math.ceil(X_train.shape[0]/batch_size), warmup_proportion=0.1, min_lr=1e-5, lr=0.001),
            #                  #                    sync_period=5, slow_step=0.5),
            #                  #metrics=['mean_squared_error'])
            #else:
            #    print("Use optimizer from the model.")
            #    model.compile(loss='mean_squared_error',
            #                   ## In this case, we cannot change the learning rate.
            #                   optimizer=model.optimizer)
            #                   #metrics=['mean_squared_error'])
            print("Use optimizer from the model.")
            model.compile(loss='mean_squared_error',optimizer=model.optimizer)

        else:

            #if use_radam == True:
            #    print("Use optimizer: %s" % ("rectified-adam"))
            #    model.compile(loss='mean_squared_error',
            #                  optimizer=Lookahead(RAdam(),sync_period=5, slow_step=0.5))
            #                  #optimizer=Lookahead(RAdam(total_steps=math.ceil(X_train.shape[0]/batch_size), warmup_proportion=0.1, min_lr=1e-5, lr=0.001),
            #                  #                    sync_period=5, slow_step=0.5),
            #                  #metrics=['mean_squared_error'])
            #else:
            #    print("Use optimizer provided by user: %s" % (optimizer_name))
            #    model.compile(loss='mean_squared_error',
            #                   ## In this case, we cannot change the learning rate.
            #                   optimizer=optimizer_name)
            #                   #metrics=['mean_squared_error'])
            print("Use optimizer provided by user: %s" % (optimizer_name))
            model.compile(loss='mean_squared_error',optimizer=optimizer_name)


    print("optimizer: %s" % (type(model.optimizer)))

    model.summary()
    # model = multi_gpu_model(model, gpus=3)

    my_callbacks = RegCallback(X_train, X_test, Y_train, Y_test, scale_para)
    # Save model
    model_chk_path = out_dir + "/best_model.hdf5"
    mcp = ModelCheckpoint(model_chk_path, save_best_only=True, save_weights_only=False,
                          verbose=1, mode='min')

    all_callbacks = list()
    all_callbacks.append(my_callbacks)
    all_callbacks.append(mcp)

    if add_ReduceLROnPlateau is True:
        print("Use ReduceLROnPlateau!")
        all_callbacks.append(keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, verbose=1,min_lr=0.000001,min_delta=0))

    if early_stop_patience is not None:
        print("Use EarlyStopping: %d" % (early_stop_patience))
        all_callbacks.append(EarlyStopping(patience=early_stop_patience,verbose=1))

    ## monitor training information
    # tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    #model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(X_test, Y_test), callbacks=[my_callbacks, mcp])
    if use_external_test_data is True:
        #model.save("debug_model.h5")
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(X_test, Y_test),
                  # callbacks=[my_callbacks, mcp])
                  callbacks=all_callbacks)
    else:
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_split=0.1,
                  # callbacks=[my_callbacks, mcp])
                  callbacks=all_callbacks)


    ## get the best model
    #if use_radam == True:
    #    model_best = load_model(model_chk_path, custom_objects = {"Lookahead": Lookahead, "RAdam":RAdam})
    #else:
    #    model_best = load_model(model_chk_path, custom_objects = {"Lookahead": Lookahead, "RAdam":RAdam})
    model_best = load_model(model_chk_path)

    y_pred = model_best.predict(X_test)

    y_pred_rev = scaling_y_rev(y_pred, scale_para)
    y_true = scaling_y_rev(Y_test, scale_para)

    #test_data['pred'] = y_pred_rev
    #test_data.to_csv("pred.csv")
    x = pd.DataFrame({"y": y_true, "y_pred": y_pred_rev.reshape(y_pred_rev.shape[0])})
    out_file = out_dir + "/" + prefix +".csv"
    print("Prediction result: %s" % (out_file))
    x.to_csv(out_file)

    res_map['model'] = model_best
    return res_map

def get_peptide_encode_method_from_model(model_file:str):
    peptide_encode_method = "one_hot"
    ## model_file is a json file
    with open(model_file, "r") as read_file:
        model_list = json.load(read_file)

    ## trained model
    if "dp_model" in model_list.keys():
        peptide_encode_method = model_list["encode_method"]
    else:
        for i in model_list.keys():
            if "encode_method" in model_list[i]:
                peptide_encode_method = model_list[i]["encode_method"]

    return peptide_encode_method


def ensemble_models_v0(input_data: str, #test_file=None,
                    models_file=None,
                    ga_file=None,
                    ensemble_method="average",
                    batch_size=64, nb_epoch=100, mod=None,
                    max_x_length=50, scale_para=None, unit="s", out_dir="./", prefix="test",
                    use_radam=False,
                    early_stop_patience=None,
                    add_reverse=False,
                    add_ReduceLROnPlateau=False):
    """
    This function is used to ensemble multiple deep learning models. It can be used for training and testing.
    """

    # print("The number of models:", len(models))

    # test data
    X_test = np.empty(1)
    Y_test = np.empty(1)

    y_pr = []
    score = []

    model_list = dict()

    ## prepare training and validation data
    train_file, test_file = split_data_file(input_data,out_dir=out_dir,test_size=0.1)

    peptide_encode_method = "one_hot"

    if ga_file is not None:

        peptide_encode_method = get_peptide_encode_method_from_model(ga_file)
        X_train, Y_train, X_test, Y_test, scale_para = data_processing(input_data=input_data, #test_file=test_file,
                                                                       mod=mod, max_x_length=max_x_length,
                                                                       scale_para=scale_para, unit=unit,
                                                                       out_dir=out_dir,add_reverse=add_reverse,
                                                                       seq_encode_method=peptide_encode_method)
        model_list['dp_model'] = dict()
        model_list['max_x_length'] = X_train.shape[1]
        model_list['aa'] = out_dir + "/aa.tsv"
        ## Useful for new data prediction
        model_list['min_rt'] = scale_para['rt_min']
        model_list['max_rt'] = scale_para['rt_max']
        if add_reverse is True:
            model_list['add_reverse'] = 1
            model_list['max_x_length'] = model_list['max_x_length']/2
        else:
            model_list['add_reverse'] = 0


        print("max_x_length: %s" % (max_x_length))
        # read models from genetic search result configure file
        optimizer_name = dict()

        with open(ga_file, "r") as read_file:
            ga_model_list = json.load(read_file)


        model_folder = os.path.dirname(ga_file)
        models = dict()
        for i in ga_model_list.keys():
            m_file = model_folder + "/" + os.path.basename(ga_model_list[i]['model'])
            print("Model file: %s -> %s" % (str(i), m_file))

            with open(m_file, "r") as json_read:
                models[i] = keras.models.model_from_json(json_read.read())

            #models[i]._layers[1].batch_input_shape = (None, X_train.shape[1], X_train.shape[2])
            #models[i]._layers[0]._batch_input_shape = (None, X_train.shape[1], X_train.shape[2])
            #models[i] = tf.keras.models.model_from_json(models[i].to_json())
            optimizer_name[i] = ga_model_list[i]['optimizer_name']

        print("Training ...")
        # For each model, train the model
        for (name, model) in models.items():
            print("Train model:", name)
            # perform sample specific training
            res_map = train_model(input_data=train_file, test_file=test_file, batch_size=batch_size,
                                  nb_epoch=nb_epoch, early_stop_patience=early_stop_patience, mod=mod,
                                  max_x_length=max_x_length, scale_para=scale_para, unit=unit,
                                  out_dir=out_dir, prefix=str(name), model=model,
                                  optimizer_name=optimizer_name[name], use_radam=use_radam, add_reverse=add_reverse,
                                  add_ReduceLROnPlateau=add_ReduceLROnPlateau,
                                  seq_encode_method=peptide_encode_method)

            ## save the model to a file:
            model_file_name = "model_" + str(name) + ".h5"
            model_file_path = out_dir + "/" + model_file_name
            res_map["model"].save(model_file_path)

            model_list['dp_model'][name] = model_file_path

            del res_map
            gc.collect()
            K.clear_session()
            #tf.reset_default_graph()

    else:

        print("Transfer learning ...")

        ## Transfer learning
        with open(models_file, "r") as read_file:
            model_list = json.load(read_file)

        model_folder = os.path.dirname(models_file)
        aa_file = os.path.basename(model_list['aa'])
        aa_file = model_folder + "/" + aa_file

        new_model_list = dict()
        new_model_list['dp_model'] = dict()
        model_i = 1

        ## peptide length check
        peptide_max_length = get_max_length_from_input_data(input_data)
        if peptide_max_length > model_list['max_x_length']:
            print("The max length (%d) in the training data should be <= the length supported by the model %d" % (peptide_max_length, model_list['max_x_length']))
            sys.exit()

        peptide_encode_method = get_peptide_encode_method_from_model(models_file)

        for (name, dp_model_file) in model_list['dp_model'].items():
            print("\nDeep learning model:", name)


            X_train, Y_train, X_test, Y_test, scale_para = data_processing(input_data=train_file, test_file=test_file,
                                                                           mod=mod,
                                                                           max_x_length=model_list['max_x_length'],
                                                                           scale_para=scale_para, unit=unit,
                                                                           out_dir=out_dir, aa_file=aa_file,
                                                                           add_reverse=add_reverse,
                                                                           random_seed=model_i,
                                                                           seq_encode_method=peptide_encode_method)

            model_i = model_i + 1
            # keras model evaluation: loss and accuracy
            # load model
            model_name = os.path.basename(dp_model_file)
            model_full_path = model_folder + "/" + model_name

            #if use_radam == True:
            #    model = load_model(model_full_path, custom_objects = {"Lookahead": Lookahead, "RAdam":RAdam})
            #else:
            #    model = load_model(model_full_path)
            model = load_model(model_full_path)
            #new_model = change_model(model, X_train.shape[1:])
            new_model = model

            print(get_peptide_length_from_model(new_model))
            if "add_reverse" in model_list.keys():
                if model_list['add_reverse'] == 1:
                    if 2*model_list['max_x_length'] != get_peptide_length_from_model(new_model):
                        print("The max length (%d) in the training data should be less than the length supported by the model %d" % (model_list['max_x_length'], get_peptide_length_from_model(new_model)))
                else:
                    if model_list['max_x_length'] != get_peptide_length_from_model(new_model):
                        print("The max length (%d) in the training data should be less than the length supported by the model %d" % (model_list['max_x_length'], get_peptide_length_from_model(new_model)))
            else:
                if model_list['max_x_length'] != get_peptide_length_from_model(new_model):
                    print("The max length (%d) in the training data should be less than the length supported by the model %d" % (model_list['max_x_length'], get_peptide_length_from_model(new_model)))


            print("Perform transfer learning ...")
            n_layers = len(new_model.layers)
            new_model.get_layer("embedding").trainable = False
            print("The number of layers: %d" % (n_layers))

            '''
            for layer in new_model.layers:
                layer_name = str(layer.name)
                if layer_name.startswith("bidirectional_"):
                    break
                else:
                    layer.trainable = False
                    print("layer (frozen:True): %s" % (layer_name))
            '''

            print(model.optimizer)
            #if use_radam == True:
            #    print("Use optimizer: %s" % ("rectified-adam"))
            #    new_model.compile(loss='mean_squared_error',
            #                      ## In this case, we cannot change the learning rate.
            #                      optimizer=Lookahead(RAdam(),sync_period=5, slow_step=0.5))
            #                      #optimizer=RAdam(),
            #                      #optimizer=Lookahead(RAdam(total_steps=1000, warmup_proportion=0.1, min_lr=1e-5, lr=0.001),
            #                      #                    sync_period=5, slow_step=0.5))
            #                      # optimizer=Adam(lr=0.0001),
            #                      # optimizer=SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=True),
            #                      #metrics=['mean_squared_error'])
            #else:
            #    print("Use optimizer: %s from saved model" % (model.optimizer.__class__.__name__))
            #    new_model.compile(loss='mean_squared_error',
            #                  ## In this case, we cannot change the learning rate.
            #                  #optimizer=model.optimizer)
            #                  optimizer=model.optimizer.__class__.__name__)
            #                  #optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.01/nb_epoch))
            #                  #optimizer=Adam(lr=0.001))
            #                  #optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True))
            #                  #metrics=['mean_squared_error'])
            print("Use optimizer: %s from saved model" % (model.optimizer.__class__.__name__))
            new_model.compile(loss='mean_squared_error',optimizer=model.optimizer.__class__.__name__)

            print("Used optimizer:")
            print(model.optimizer)
            my_callbacks = RegCallback(X_train, X_test, Y_train, Y_test, scale_para)
            # Save model
            model_chk_path = out_dir + "/best_model.hdf5"
            if os.path.exists(model_chk_path):
                os.remove(model_chk_path)

            mcp = ModelCheckpoint(model_chk_path, save_best_only=True,
                                  save_weights_only=False,
                                  verbose=1, mode='min')

            all_callbacks = list()
            all_callbacks.append(my_callbacks)
            all_callbacks.append(mcp)

            if add_ReduceLROnPlateau is True:
                print("Use ReduceLROnPlateau!")
                all_callbacks.append(
                    keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1, min_lr=0.00001, min_delta=0))

            if early_stop_patience is not None:
                print("Use EarlyStopping: %d" % (early_stop_patience))
                all_callbacks.append(EarlyStopping(patience=early_stop_patience, verbose=1))

            #all_callbacks.append(LearningRateScheduler(PolynomialDecay(maxEpochs=nb_epoch, initAlpha=0.001, power=5)))

            ## monitor training information
            # tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
            new_model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(X_test, Y_test),
                      callbacks=all_callbacks)#, keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1, min_lr=0.000001)])

            ## get the best model
            #if use_radam == True:
            #    best_model = load_model(model_chk_path, custom_objects = {"Lookahead": Lookahead, "RAdam":RAdam})
            #else:
            #    best_model = load_model(model_chk_path)
            best_model = tf.keras.models.load_model(model_chk_path)
            ## save the model to a file:
            model_file_name = "model_" + str(name) + ".h5"
            model_file_path = out_dir + "/" + model_file_name
            best_model.save(model_file_path)

            new_model_list['dp_model'][name] = model_file_path

            gc.collect()
            K.clear_session()
            #tf.reset_default_graph()

        new_model_list['max_x_length'] = model_list['max_x_length']
        new_aa_file = out_dir + "/" + os.path.basename(model_list['aa'])
        copyfile(aa_file, new_aa_file)
        new_model_list['aa'] = new_aa_file

        ## Useful for new data prediction
        #new_model_list['min_rt'] = scale_para['rt_min']
        #new_model_list['max_rt'] = scale_para['rt_max']

        model_list = new_model_list


    # save model data
    #file_all_models = open(out_dir + "/all_models.obj", 'wb')
    #pickle.dump(models, file_all_models)
    #file_all_models.close()

    ####################################################################################################################
    print("Ensemble learning ...")


    para = dict()
    para['rt_min'] = scale_para['rt_min']
    para['rt_max'] = scale_para['rt_max']
    para['scaling_method'] = scale_para['scaling_method']

    model_list['rt_min'] = scale_para['rt_min']
    model_list['rt_max'] = scale_para['rt_max']
    model_list['scaling_method'] = scale_para['scaling_method']

    model_list['encode_method'] = peptide_encode_method



    if scale_para['scaling_method'] == "mean_std":
        para['rt_mean'] = scale_para['rt_mean']
        para['rt_std'] = scale_para['rt_std']
        model_list['rt_mean'] = scale_para['rt_mean']
        model_list['rt_std'] = scale_para['rt_std']

    elif scale_para['scaling_method'] == "single_factor":
        para['scaling_factor'] = scale_para['scaling_factor']
        model_list['scaling_factor'] = scale_para['scaling_factor']

    if add_reverse is True:
        model_list['add_reverse'] = 1
        para['add_reverse'] = 1
    else:
        model_list['add_reverse'] = 0
        para['add_reverse'] = 0

    ## save result
    model_json = out_dir + "/model_all.json"
    with open(model_json, 'w') as f:
        json.dump(model_list, f)

    ## best model combination selection
    pred_all, y_true = rt_predict(model_file=model_json, test_file=test_file, out_dir=out_dir, prefix=prefix)
    pred_all = pd.DataFrame(pred_all)
    best_i, best_metric, model_data = model_selection(pred_all,y_true,para=para,metric="r2")
    model_list["best_models"] = ",".join([str(i) for i in best_i[0]])
    model_list["best_performance"] = best_metric
    model_data = pd.DataFrame(model_data)
    model_data.to_csv(out_dir+"/models_combination_metrics.tsv",sep="\t",index=False)

    ## save result
    model_json = out_dir + "/model.json"
    with open(model_json, 'w') as f:
        json.dump(model_list, f)
    ####################################################################################################################


def get_aa_file(model_file):
    model_folder = os.path.dirname(model_file)
    aa_file = model_folder + "/aa.tsv"
    if not os.path.isfile(aa_file):
        print("Amino acid file not found:%s" % (aa_file))
        sys.exit()

    return aa_file


def use_one_model(model_file, new_model_file, use_model_id="0"):
    with open(model_file, "r") as read_file:
        model_list = json.load(read_file)
    selected_model = {use_model_id: model_list['dp_model'][use_model_id]}
    model_list['dp_model'] = selected_model
    with open(new_model_file, 'w') as f:
        json.dump(model_list, f, indent=2)


def use_selected_model(model_file, use_model_id="0"):
    with open(model_file, "r") as read_file:
        model_list = json.load(read_file)
    selected_model = {use_model_id: model_list['dp_model'][use_model_id]}
    model_list['dp_model'] = selected_model
    return model_list


def two_step_ensemble_models(models_file:str, input_data:str,
                    ensemble_method="average",
                    batch_size=64, nb_epoch=100,
                    scale_para=None, unit="s",
                    out_dir="./",
                    prefix="test",
                    early_stop_patience=0,
                    add_reverse=False,
                    add_ReduceLROnPlateau=False,
                    gpu_device=None,
                    do_evaluation_after_each_epoch=False,
                    outlier_ratio=0.03,
                    max_x_length=0):
    # step 1: training
    print("Step 1:\n")
    tmp_model_dir = out_dir + "/step1"
    if not os.path.exists(tmp_model_dir):
        os.makedirs(tmp_model_dir)

    trained_model_file = ensemble_models(models_file=models_file, input_data=input_data, ensemble_method=ensemble_method,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    scale_para=scale_para, unit=unit,
                    out_dir=tmp_model_dir, prefix=prefix,
                    early_stop_patience=early_stop_patience,
                    add_reverse=add_reverse,
                    add_ReduceLROnPlateau=add_ReduceLROnPlateau,
                    gpu_device=gpu_device,
                    do_evaluation_after_each_epoch=do_evaluation_after_each_epoch,
                    select_best_models=False,
                    use_model_id="0",
                    max_x_length=max_x_length)
    tmp_prediction_dir = out_dir + "/step1/prediction/"
    if not os.path.exists(tmp_prediction_dir):
        os.makedirs(tmp_prediction_dir)

    rt_predict(model_file=tmp_model_dir + "/model.json", test_file=input_data, out_dir=tmp_prediction_dir,
               prefix=prefix, gpu_device=gpu_device)
    a = pd.read_csv(tmp_prediction_dir+"/"+str(prefix)+".tsv", sep="\t", low_memory=False)
    a['rt_error'] = np.abs(a['y'] - a['y_pred'])
    a.sort_values(by=['rt_error'], inplace=True, ascending=False)
    top_n = math.ceil(a.shape[0] * outlier_ratio)
    a_top = a.head(top_n)
    a.to_csv(tmp_prediction_dir + "/pred_sorted.tsv", sep="\t", index=False)
    a_top.to_csv(tmp_prediction_dir+"/top_n_outlier.tsv", sep="\t", index=False)
    all_train = pd.read_csv(input_data, sep="\t", low_memory=False)
    new_all_train = all_train[~all_train['x'].isin(list(a_top['x']))]
    new_train_data_file = out_dir + "/new_train_data.tsv"
    new_all_train.to_csv(new_train_data_file, sep="\t", index=False)

    # step 2: training
    print("Step 2:\n")
    ensemble_models(models_file=models_file, input_data=new_train_data_file, ensemble_method=ensemble_method,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    scale_para=scale_para, unit=unit,
                    out_dir=out_dir, prefix=prefix,
                    early_stop_patience=early_stop_patience,
                    add_reverse=add_reverse,
                    add_ReduceLROnPlateau=add_ReduceLROnPlateau,
                    gpu_device=gpu_device,
                    do_evaluation_after_each_epoch=do_evaluation_after_each_epoch,
                    max_x_length=max_x_length)


def ensemble_models(models_file:str, input_data:str, #test_file=None,
                    ensemble_method="average",
                    batch_size=64, nb_epoch=100, #mod=None,
                    max_x_length=0,
                    scale_para=None, unit="s",
                    out_dir="./",
                    prefix="test",
                    #use_radam=False,
                    early_stop_patience=0,
                    add_reverse=False,
                    add_ReduceLROnPlateau=False,
                    gpu_device=None,
                    do_evaluation_after_each_epoch=False,
                    select_best_models=True,
                    use_model_id=None):
    """
    This function is used to ensemble multiple deep learning models. It can be used for training and testing.
    """
    with open(models_file, "r") as read_file:
        model_list = json.load(read_file)

    if model_list['max_x_length'] < max_x_length:
        new_model_dir = out_dir + "/new_base_model_dir"
        model_update = ModelUpdate(out_dir=new_model_dir,max_x_length=max_x_length)
        with Pool(1) as p:
            p_res = p.map(model_update.update_model_input_shape, [models_file])
        models_file = p_res[0]
        with open(models_file, "r") as read_file:
            model_list = json.load(read_file)
    else:
        max_x_length = model_list['max_x_length']

    if use_model_id is not None:
        model_list = use_selected_model(models_file,use_model_id=use_model_id)

    peptide_encode_method = get_peptide_encode_method_from_model(models_file)
    aa_file = get_aa_file(models_file)

    # prepare training and validation data
    train_file, test_file = split_data_file(input_data, out_dir=out_dir, test_size=0.1)
    X_train, Y_train, X_test, Y_test, scale_para = data_processing(input_data=train_file, test_file=test_file,
                                                                   max_x_length=max_x_length,
                                                                   scale_para=scale_para, unit=unit,
                                                                   out_dir=out_dir,
                                                                   aa_file=aa_file,
                                                                   add_reverse=add_reverse,
                                                                   random_seed=2021,
                                                                   seq_encode_method=peptide_encode_method)

    new_model_list = dict()
    new_model_list['dp_model'] = dict()

    # peptide length check
    peptide_max_length = get_max_length_from_input_data(input_data)
    if peptide_max_length > model_list['max_x_length']:
        print("The max length (%d) in the training data should be <= the length supported by the model %d" % (
        peptide_max_length, model_list['max_x_length']))
        sys.exit()

    if gpu_device is not None:
        gpu_device = str(gpu_device)
        gpus_ids = gpu_device.split(",")
        print("Use GPU:")
        print(gpus_ids)

        model_tasks = list()

        n_total_models = 0
        model_name_list = list()
        model_file_path = list()
        for (name, dp_model_file) in model_list['dp_model'].items():
            n_total_models = n_total_models + 1
            model_name_list.append(name)
            model_file_path.append(dp_model_file)

        n_total_models = len(model_list['dp_model'])
        n_per_task = math.ceil(1.0 * n_total_models / len(gpus_ids))
        print("Total GPUs: %d" % (len(gpus_ids)))
        print("GPU device: %s" % (gpus_ids))
        print("Tasks on each GPU: %d" % (n_per_task))

        for i in range(len(gpus_ids)):
            model_tasks.append(ModelT(models_file, out_dir))
            model_tasks[i].add_gpu_device(gpus_ids[i])
            model_tasks[i].set_scale_para(scale_para)
            model_tasks[i].add_train_data(X_train, Y_train, X_test, Y_test)
            model_tasks[i].set_batch_size(batch_size)
            model_tasks[i].set_epoch(nb_epoch)
            model_tasks[i].set_ReduceLROnPlateau(add_ReduceLROnPlateau)
            model_tasks[i].set_early_stop_patience(early_stop_patience)
            model_tasks[i].set_do_evaluation_after_each_epoch(do_evaluation_after_each_epoch)
            for j in range(n_per_task):
                if len(model_name_list) >= 1:
                    model_tasks[i].add_model(model_name_list.pop(0), model_file_path.pop(0))

        with Pool(len(gpus_ids)) as p:
            trained_model = p.map(run_model_t, model_tasks)
    else:
        print("Use one GPU or no GPU!")
        model_t = ModelT(models_file, out_dir)
        model_t.set_scale_para(scale_para)
        model_t.add_train_data(X_train, Y_train, X_test, Y_test)
        model_t.set_batch_size(batch_size)
        model_t.set_epoch(nb_epoch)
        model_t.set_ReduceLROnPlateau(add_ReduceLROnPlateau)
        model_t.set_early_stop_patience(early_stop_patience)
        model_t.set_do_evaluation_after_each_epoch(do_evaluation_after_each_epoch)
        for (name, dp_model_file) in model_list['dp_model'].items():
            model_t.add_model(name, dp_model_file)
        with Pool(1) as p:
            trained_model = p.map(run_model_t, [model_t])

    for model_t in trained_model:
        for name in model_t.keys():
            new_model_list["dp_model"][name] = model_t[name]

    new_model_list['max_x_length'] = model_list['max_x_length']
    new_aa_file = out_dir + "/" + os.path.basename(model_list['aa'])
    copyfile(aa_file, new_aa_file)
    new_model_list['aa'] = new_aa_file

    model_list = new_model_list

    ####################################################################################################################
    print("Ensemble learning ...")

    para = dict()
    para['rt_min'] = scale_para['rt_min']
    para['rt_max'] = scale_para['rt_max']
    para['scaling_method'] = scale_para['scaling_method']

    model_list['rt_min'] = scale_para['rt_min']
    model_list['rt_max'] = scale_para['rt_max']
    model_list['scaling_method'] = scale_para['scaling_method']

    model_list['encode_method'] = peptide_encode_method

    if scale_para['scaling_method'] == "mean_std":
        para['rt_mean'] = scale_para['rt_mean']
        para['rt_std'] = scale_para['rt_std']
        model_list['rt_mean'] = scale_para['rt_mean']
        model_list['rt_std'] = scale_para['rt_std']

    elif scale_para['scaling_method'] == "single_factor":
        para['scaling_factor'] = scale_para['scaling_factor']
        model_list['scaling_factor'] = scale_para['scaling_factor']

    if add_reverse is True:
        model_list['add_reverse'] = 1
        para['add_reverse'] = 1
    else:
        model_list['add_reverse'] = 0
        para['add_reverse'] = 0

    if select_best_models:
        # save result
        print("Select best models ...")
        model_json = out_dir + "/model_all.json"
        with open(model_json, 'w') as f:
            json.dump(model_list, f, indent=2)

        # best model combination selection
        pred_all, y_true = rt_predict(model_file=model_json, test_file=test_file, out_dir=out_dir, prefix=prefix,
                                      gpu_device=gpu_device)
        pred_all = pd.DataFrame(pred_all)
        best_i, best_metric, model_data = model_selection(pred_all,y_true,para=para,metric="median_absolute_error")
        model_list["best_models"] = ",".join([str(i) for i in best_i[0]])
        model_list["best_performance"] = best_metric
        model_data = pd.DataFrame(model_data)
        model_data.to_csv(out_dir+"/models_combination_metrics.tsv", sep="\t", index=False)

    # save result
    model_json = out_dir + "/model.json"
    with open(model_json, 'w') as f:
        json.dump(model_list, f, indent=2)

    return model_json
    ####################################################################################################################


def train_with_multiple_gpus(model_file:str,x_train_file:str,y_train_file:str,x_test_file:str,y_test_file:str,para,out_dir="./"):
    # for each
    print("TODO")


def change_model(model, new_input_shape):
    # replace input shape of first layer
    """
    Used by AutoRT
    :param model:
    :param new_input_shape:
    :return:
    """
    print("Base model ...")
    print(model.get_weights())
    #model._layers[1].batch_input_shape = (None,new_input_shape[0],new_input_shape[1])
    model._layers[0]._batch_input_shape = (None, new_input_shape[0], new_input_shape[1])

    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(model.to_json())


    # copy weights from old model to new one
    for layer in new_model.layers:
        try:
            print("layer: %s" % (layer.name))
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    new_model.compile(loss='mean_squared_error',
                  ## In this case, we cannot change the learning rate.
                  optimizer=model.optimizer)
                  #metrics=['mean_squared_error'])

    new_model.summary()

    print("New model ...")
    print(new_model.get_weights())
    return new_model

def change_model_input_shape(model, new_input_shape):
    model._layers[0]._batch_input_shape = (None, new_input_shape[0], new_input_shape[1])
    return model

def get_model_input_shape(model):
    return model._layers[0]._batch_input_shape

def get_peptide_length_from_model(model):
    return model.input_shape[1]
    #return model._layers[0]._batch_input_shape[1]


def ensemble_predict(model_file:str, x, y=None, para=None, out_dir="./", method="average",batch_size=64, prefix="test", use_radam=False):

    res_to_file = np.empty(0)
    ## prediction result
    y_pr_final = np.empty(0)

    if method == "average":
        print("Average ...")
        res = dl_models_predict(model_file, x=x, y=y, batch_size=batch_size,out_dir=out_dir,prefix=prefix,use_radam=use_radam)
        y_pr_final = res.mean(axis=1)
        #np.save("res", res)
        #np.save("y_pr_final",y_pr_final)
        res_to_file = res
        res_to_file = np.append(res_to_file, y_pr_final.reshape([y_pr_final.shape[0],1]), axis=1)

    if y is not None:
        # evaluate the final _result
        # ROC
        print("\n\nFinal model:")
        if len(y.shape) >= 2:
            y_true_class = np.argmax(y, axis=1)
        else:
            y_true_class = y

        # classification report: precision recall f1-score support
        # y_class_final = np.where(y_pr_final > 0.5, 1, 0) #np.argmax(y_pr_final, axis=1)
        out_prefix = prefix + "_final"
        evaluate_model(y_true_class, y_pr_final, para=para, out_dir=out_dir, prefix=out_prefix)

        #np.save("y",y)
        #res_to_file = np.append(res_to_file, y_true_class.reshape([y_true_class.shape[0],1]), axis=1)

    #np.save("final_res",res_to_file)
    if y is not None:
        return res_to_file
    else:
        return y_pr_final

def rt_predict(model_file:str, test_file:str, out_dir="./", prefix="test", method = "average", use_radam=False, gpu_device=None):

    if gpu_device is not None:
        print("Use GPU device: %s" % (str(gpu_device)))
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)

    res_to_file = np.empty(0)
    ## prediction result
    y_pr_final = np.empty(0)

    y_true = None

    peptide_encode_method = get_peptide_encode_method_from_model(model_file)

    if method == "average":
        print("Average ...")

        with open(model_file, "r") as read_file:
            model_list = json.load(read_file)

        x_predict_data = processing_prediction_data(model_file, test_file, seq_encode_method=peptide_encode_method)

        input_data = pd.read_table(test_file, sep="\t", header=0, low_memory=False)

        if "y" in input_data.columns.values:
            y_true = scaling_y(np.asarray(input_data['y']), model_list)
            ## prediction data for each model
            pred_all = dl_models_predict(model_file, x=x_predict_data, y=y_true, out_dir=out_dir, prefix=prefix, use_radam=use_radam)
        else:
            ## prediction data for each model
            pred_all = dl_models_predict(model_file, x=x_predict_data, out_dir=out_dir, prefix=prefix, use_radam=use_radam)
        #y_pr_final = res.mean(axis=1)
        #y_pr_final = np.median(res,axis=1)
        #np.save("res",pred_all)
        #np.save("y_true",y_true)
        rt_pred = np.apply_along_axis(combine_rts, 1, pred_all, reverse=True, scale_para=model_list, method="mean", remove_outlier=True)
        #y_pr_final = np.apply_along_axis(combine_rts, 1, res, reverse=False, method="mean", remove_outlier=True)
        #y_pr_final = np.apply_along_axis(combine_rts, 1, res, reverse=False, method="median", remove_outlier=False)
        #y_pr_final = np.apply_along_axis(combine_rts, 1, res, reverse=False, method="median", remove_outlier=False)

        #np.save("res", res)
        #np.save("y_pr_final", y_pr_final)
        #res_to_file = res
        #res_to_file = np.append(res_to_file, y_pr_final.reshape([y_pr_final.shape[0], 1]), axis=1)



        # rt_pred = minMaxScoreRev(y_pr_final, model_list['min_rt'], model_list['max_rt'])


        input_data['y_pred'] = rt_pred

        ## output
        out_file = out_dir + "/" + prefix + ".tsv"
        input_data.to_csv(out_file,sep="\t",index=False)

        ## evaluate
        if "y" in input_data.columns.values:
            #y_true = minMaxScale(np.asarray(input_data['y']), model_list['min_rt'], model_list['max_rt'])
            #y_pr_final = minMaxScale(np.asarray(input_data['y']), model_list['min_rt'], model_list['max_rt'])
            out_prefix = prefix + "_" + "evaluate"
            evaluate_model(input_data['y'], rt_pred, para=model_list, out_dir=out_dir, prefix=out_prefix, reverse=False)

        return [pred_all, y_true]


def dl_models_predict(model_file, x, y=None,batch_size=2048, out_dir="./", prefix="test", use_radam=False):

    with open(model_file, "r") as read_file:
        model_list = json.load(read_file)


    y_dp = np.zeros(0)

    model_folder = os.path.dirname(model_file)

    if x.shape[0] < batch_size:
        # remove a warning message
        batch_size = 64
    #if x.shape[0] < 64:
    #    batch_size = 2
    #print(batch_size)
    #avg_models = list()
    for (name, dp_model_file) in model_list['dp_model'].items():
        print("\nDeep learning model:", name)

        if "best_models" in model_list:
            best_models = model_list['best_models'].split(",")
            if name not in best_models:
                continue
        # keras model evaluation: loss and accuracy
        # load model
        model_name = os.path.basename(dp_model_file)
        model_full_path = model_folder + "/" + model_name

        #if use_radam == True:
        #    model = load_model(model_full_path,custom_objects = {"Lookahead": Lookahead, "RAdam":RAdam})
        #else:
        #    model = load_model(model_full_path,custom_objects = {"Lookahead": Lookahead, "RAdam":RAdam})

        pred_np_file = out_dir + "/" + str(prefix) + "_" + str(name) + ".npy"
        p = Process(target=do_prediction, args=(model_full_path,x,y,batch_size,pred_np_file))
        p.start()
        p.join()
        y_prob = np.load(pred_np_file)
        #model = tf.keras.models.load_model(model_full_path)

        #avg_models.append(model)

        #print("batch size: "+str(batch_size))

        #y_prob = model.predict(x,batch_size=batch_size)
        ## for class 1
        #y_prob_dp_vector = y_prob[:, 1]
        y_prob_dp_vector = y_prob
        y_prob_dp = y_prob_dp_vector.reshape([y_prob_dp_vector.shape[0], 1])
        if y_dp.shape[0] != 0:
            y_dp = np.append(y_dp, y_prob_dp, axis=1)
        else:
            y_dp = y_prob_dp

        if y is not None:
            #evaluation_res = model.evaluate(x, y)
            #print("Metrics:")
            #print(evaluation_res)

            # ROC
            if len(y.shape) >= 2:
                y_true_class = np.argmax(y, axis=1)
            else:
                y_true_class= y
            out_prefix = prefix + "_" + str(name)
            evaluate_model(y_true_class, y_prob_dp_vector, para=model_list, out_dir=out_dir, prefix=out_prefix)

        #del model
        gc.collect()
        K.clear_session()
        #tf.reset_default_graph()

    return y_dp


def do_prediction(model_full_path,x,y,batch_size,out_file):
    model = tf.keras.models.load_model(model_full_path)
    y_prob = model.predict(x, batch_size=batch_size)
    np.save(out_file, y_prob)
    with open(out_file, 'wb') as f:
        np.save(f, y_prob)
    if y is not None:
        evaluation_res = model.evaluate(x, y)
        print("Metrics:")
        print(evaluation_res)