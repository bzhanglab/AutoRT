
from numpy.random import seed
seed(2019)
#from tensorflow import set_random_seed
#set_random_seed(2020)
import matplotlib
matplotlib.use("agg")
import argparse
import sys
from autort.RTModels import ensemble_models, rt_predict, two_step_ensemble_models
import os
import time


def main():

    if len(sys.argv) == 1:
        print("python autort.py [train, predict]")
        sys.exit(0)
    else:

        mode = sys.argv[1]

        if mode == "train":
            start_time = time.time()
            ## general training and training for transfer learning
            parser = argparse.ArgumentParser(
                description='AutoRT')
            parser.add_argument('-i', '--input', default=None, type=str, required=True,
                                help="Input data for training")
            #parser.add_argument('-t', '--test', default=None, type=str,
            #                    help="Input data for testing")

            parser.add_argument('-o', '--out_dir', default="./", type=str,
                                help="Output directory")

            parser.add_argument('-e', '--epochs', default=40, type=int,help="The number of epochs, default is 40.")
            parser.add_argument('-b', '--batch_size', default=64, type=int,help="Batch size for training, default is 64.")
            parser.add_argument('-r2', '--max_rt', default=0, type=int,help="The maximum retention time. If the value is 0 (default), the maximum retention time will be automatically infered from the input training data.")
            parser.add_argument('-l', '--max_length', default=0, type=int,help="The length of the longest peptide to consider for modeling. If the value is 0 (default), it will be automatically infered from the input model file.")
            #parser.add_argument('-p', '--mod', default=None, type=str,help="The integer number(s) used to represent modified amino acid(s) in training data. For example, if use 1 to represent M with oxidation and use 2 to represent S with phosphorylation, then the setting is '-p 1,2'")
            parser.add_argument('-u', '--unit', default="m", type=str,help="The unit of retention time in training data, s: second, m: minute (default).")
            parser.add_argument('-sm', '--scale_method', default="min_max", type=str,help="Scaling method for RT tranformation: min_max (default), mean_std and single_factor. This is used in training. Default is 'min_max'. The default method works well in most of cases. This should not be changed unless users know well about the meaning of these methods.")
            parser.add_argument('-sf', '--scale_factor', default=None, type=float,help="This is only useful when 'single_factor' is set for '-sm'.")

            parser.add_argument('-m', '--model_file', default=None, type=str,help="Trained model file. Only useful when perform transfer learning and RT prediction.")
            #parser.add_argument('-g', '--ga', default=None, type=str,help="Model configuration file. Only useful when train models from scratch.")
            #parser.add_argument('-w', '--top_n', default=10, type=int)

            #parser.add_argument('-a', '--radam', action='store_true')
            parser.add_argument('-r', '--add_reverse', action='store_true',help="Add reversed peptide in peptide encoding. This parameter will be removed in a future version.")

            parser.add_argument('-n', '--early_stop_patience', default=None, type=int,help="Number of epochs with no improvement after which training will be stopped.")

            # add_ReduceLROnPlateau
            parser.add_argument('-rlr', '--add_ReduceLROnPlateau', action='store_true',help="Reduce learning rate when a metric has stopped improving.")

            parser.add_argument('-g', '--gpu', default=None, type=str,help="Set gpu IDs that can be used. For example: 1,2,3.")
            parser.add_argument('-d', '--do_evaluation_after_each_epoch', action='store_true',help="Do evaluation after each epoch during model training.")
            parser.add_argument('-f', '--outlier_ratio', default=0.03, type=float,
                                help="The percentage of data (outlier) will be removed from the training data using a two-step training.")

            args = parser.parse_args(sys.argv[2:len(sys.argv)])

            input_file = args.input
            #test_file = args.test
            out_dir = args.out_dir
            max_rt = args.max_rt
            #mod = args.mod
            unit = args.unit
            gpu_ids = args.gpu
            max_x_length = args.max_length

            #use_radam = args.radam

            #if mod is not None:
            #    mod = mod.split(",")

            epochs = args.epochs
            batch_size = args.batch_size

            model_file = args.model_file
            #ga = args.ga

            if args.early_stop_patience is None:
                early_stop_patience = 0
            else:
                early_stop_patience = int(args.early_stop_patience)

            add_reverse = args.add_reverse
            add_ReduceLROnPlateau = args.add_ReduceLROnPlateau

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            scale_para = dict()
            scale_para['scaling_method'] = args.scale_method
            scale_para['rt_max'] = max_rt

            do_evaluation_after_each_epoch = args.do_evaluation_after_each_epoch

            scale_factor = args.scale_factor
            if scale_factor is not None:
                scale_para['scaling_factor'] = scale_factor

            print("Scaling method: %s" % (str(scale_para['scaling_method'])))

            outlier_ratio = args.outlier_ratio
            if outlier_ratio > 0:
                two_step_ensemble_models(input_data=input_file, nb_epoch=epochs, batch_size=batch_size,
                                         scale_para=scale_para, unit=unit, models_file=model_file,
                                         out_dir=out_dir, early_stop_patience=early_stop_patience,
                                         add_reverse=add_reverse, add_ReduceLROnPlateau=add_ReduceLROnPlateau,
                                         gpu_device=gpu_ids,
                                         do_evaluation_after_each_epoch=do_evaluation_after_each_epoch,
                                         outlier_ratio=outlier_ratio,
                                         max_x_length=max_x_length)
            else:
                ensemble_models(input_data=input_file, nb_epoch=epochs, batch_size=batch_size,
                                scale_para=scale_para,  unit=unit, models_file=model_file,
                                out_dir=out_dir, early_stop_patience=early_stop_patience,
                                add_reverse=add_reverse, add_ReduceLROnPlateau=add_ReduceLROnPlateau,
                                gpu_device=gpu_ids, do_evaluation_after_each_epoch=do_evaluation_after_each_epoch,
                                max_x_length=max_x_length)

            end_time = time.time()
            print("Total time used: %f minutes" % ((end_time - start_time)/60.0))

        elif mode == "predict":

            parser = argparse.ArgumentParser(
                description='AutoRT')

            parser.add_argument('-t', '--test', default=None, type=str,
                                help="Input data for testing")
            parser.add_argument('-o', '--out_dir', default="./", type=str,
                                help="Output directory")
            parser.add_argument('-p', '--prefix', default="test", type=str)
            parser.add_argument('-s', '--ensemble', default=None, type=str)
            parser.add_argument('-a', '--radam', action='store_true')

            args = parser.parse_args(sys.argv[2:len(sys.argv)])

            test_file = args.test
            out_dir = args.out_dir
            prefix = args.prefix

            use_radam = args.radam


            ensemble = args.ensemble

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            rt_predict(model_file=ensemble,test_file=test_file, out_dir=out_dir, prefix=prefix, use_radam=use_radam)


if __name__=="__main__":
    main()






