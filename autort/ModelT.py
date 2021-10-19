import random

import tensorflow as tf
import json
import os
from .RegCallback import RegCallback
import gc
import tensorflow.keras.backend as K
import time
import numpy as np


def run_model_t(model_t):
    model_t.run()
    return model_t.trained_model


def get_peptide_length_from_model(model):
    return model.input_shape[1]


class ModelT:

    def __init__(self, models_file:str, out_dir="./"):
        # This is a json file which contains model files.
        self.models_file = models_file
        self.out_dir = out_dir

        self.model_dict = dict()
        self.gpu_id = None
        self.trained_model = dict()

        # parameters
        # parameters used by all instances
        # do evaluation after each epoch
        self.do_evaluation_after_each_epoch = False
        self.add_ReduceLROnPlateau = False
        self.early_stop_patience = 0
        self.batch_size = 64
        self.n_epoch = 40
        self.scale_para = dict()
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def set_do_evaluation_after_each_epoch(self, x: bool):
        self.do_evaluation_after_each_epoch = x

    def set_ReduceLROnPlateau(self, x: bool):
        self.add_ReduceLROnPlateau = x

    def set_early_stop_patience(self,x: int):
        self.early_stop_patience = x

    def set_batch_size(self,x: int):
        self.batch_size = x

    def set_epoch(self,x: int):
        self.n_epoch = x

    def set_scale_para(self,x: dict):
        self.scale_para = x

    def add_train_data(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def add_model(self,model_name:str,model_file:str):
        self.model_dict[model_name] = model_file

    def add_gpu_device(self,gpu_id=None):
        print("Use GPU: %s" % (gpu_id))
        self.gpu_id = gpu_id

    def run(self):

        tf.random.set_seed(2021)
        np.random.seed(2021)
        random.seed(2021)

        tmp_dir = self.out_dir + "/tmp/model_" + "".join(self.model_dict.keys())
        os.makedirs(tmp_dir,exist_ok=True)

        if self.gpu_id is not None:
            print("Use GPU device: %s" % (str(self.gpu_id)))
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        with open(self.models_file,"r") as f:
            model_list = json.load(f)

        model_folder = os.path.dirname(self.models_file)
        aa_file = os.path.basename(model_list['aa'])
        aa_file = model_folder + "/" + aa_file

        for (name, dp_model_file) in self.model_dict.items():
            start_time = time.time()
            if self.gpu_id is not None:
                print("\nModel training: %s on GPU %s" % (name, self.gpu_id))
            else:
                print("\nModel training: %s" % (name))

            model_name = os.path.basename(dp_model_file)
            model_full_path = model_folder + "/" + model_name
            model = tf.keras.models.load_model(model_full_path)
            new_model = model

            print(get_peptide_length_from_model(new_model))
            if "add_reverse" in model_list.keys():
                if model_list['add_reverse'] == 1:
                    if 2 * model_list['max_x_length'] != get_peptide_length_from_model(new_model):
                        print(
                            "The max length (%d) in the training data should be less than the length supported by the model %d" % (
                            model_list['max_x_length'], get_peptide_length_from_model(new_model)))
                else:
                    if model_list['max_x_length'] != get_peptide_length_from_model(new_model):
                        print(
                            "The max length (%d) in the training data should be less than the length supported by the model %d" % (
                            model_list['max_x_length'], get_peptide_length_from_model(new_model)))
            else:
                if model_list['max_x_length'] != get_peptide_length_from_model(new_model):
                    print(
                        "The max length (%d) in the training data should be less than the length supported by the model %d" % (
                        model_list['max_x_length'], get_peptide_length_from_model(new_model)))

            print("Perform transfer learning ...")
            n_layers = len(new_model.layers)
            # new_model.get_layer("embedding").trainable = False
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
            print("Use optimizer: %s from saved model" % (model.optimizer.__class__.__name__))
            new_model.compile(loss='mean_squared_error', optimizer=model.optimizer.__class__.__name__)

            print("Used optimizer:")
            print(model.optimizer)
            all_callbacks = list()
            if self.do_evaluation_after_each_epoch is True:
                my_callbacks = RegCallback(self.x_train, self.x_test, self.y_train, self.y_test, self.scale_para)
                all_callbacks.append(my_callbacks)
            # Save model
            model_chk_path = tmp_dir + "/best_model.hdf5"
            if os.path.exists(model_chk_path):
                os.remove(model_chk_path)

            mcp = tf.keras.callbacks.ModelCheckpoint(model_chk_path, save_best_only=True, save_weights_only=False, verbose=1, mode='min')
            all_callbacks.append(mcp)

            if self.add_ReduceLROnPlateau is True:
                print("Use ReduceLROnPlateau!")
                # all_callbacks.append(tf.keras.callbacks.LearningRateScheduler(self.scheduler))
                all_callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.8, verbose=1, min_lr=0.00001, min_delta=0,
                                     monitor="val_loss",mode="min"))

            if self.early_stop_patience > 0:
                print("Use EarlyStopping: %d" % (self.early_stop_patience))
                all_callbacks.append(tf.keras.callbacks.EarlyStopping(patience=self.early_stop_patience, verbose=1))

            # all_callbacks.append(LearningRateScheduler(PolynomialDecay(maxEpochs=nb_epoch, initAlpha=0.001, power=5)))

            # monitor training information
            # tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
            print("Batch size: %d" % (self.batch_size))
            print("Epoch: %d" % (self.n_epoch))
            new_model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.n_epoch,
                          validation_data=(self.x_test, self.y_test),
                          callbacks=all_callbacks,verbose=0)  # , keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1, min_lr=0.000001)])

            best_model = tf.keras.models.load_model(model_chk_path)
            ## save the model to a file:
            model_file_name = "model_" + str(name) + ".h5"
            model_file_path = self.out_dir + "/" + model_file_name
            best_model.save(model_file_path)

            self.trained_model[name] = model_file_path

            gc.collect()
            K.clear_session()

            end_time = time.time()
            print("\nModel training: %s finished, time used: %f minutes" % (name, (end_time-start_time)/60))

    def scheduler(epoch:int, lr):
        if epoch < 10:
            return lr
        else:
            return lr * 0.95