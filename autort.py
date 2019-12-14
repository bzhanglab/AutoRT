
from numpy.random import seed
seed(2019)
from tensorflow import set_random_seed
set_random_seed(2020)
import matplotlib
matplotlib.use("agg")
import argparse
import sys
from autort.RTModels import ensemble_models, rt_predict
import os


def main():

    if len(sys.argv) == 1:
        print("python autort.py [train, predict]")
        sys.exit(0)
    else:

        mode = sys.argv[1]

        if mode == "train":

            ## general training and training for transfer learning
            parser = argparse.ArgumentParser(
                description='AutoRT')
            parser.add_argument('-i', '--input', default=None, type=str, required=True,
                                help="Input data for training")
            parser.add_argument('-t', '--test', default=None, type=str,
                                help="Input data for testing")

            parser.add_argument('-o', '--out_dir', default="./", type=str,
                                help="Output directory")

            parser.add_argument('-e', '--epochs', default=20, type=int)
            parser.add_argument('-b', '--batch_size', default=128, type=int)
            parser.add_argument('-r2', '--max_rt', default=0, type=int)
            parser.add_argument('-l', '--max_length', default=0, type=int)
            parser.add_argument('-p', '--mod', default=None, type=str)
            parser.add_argument('-u', '--unit', default="s", type=str)
            parser.add_argument('-sm', '--scale_method', default="min_max", type=str)
            parser.add_argument('-sf', '--scale_factor', default=None, type=float)

            parser.add_argument('-m', '--model_file', default=None, type=str)
            parser.add_argument('-g', '--ga', default=None, type=str)
            #parser.add_argument('-w', '--top_n', default=10, type=int)

            parser.add_argument('-a', '--radam', action='store_true')
            parser.add_argument('-r', '--add_reverse', action='store_true')

            parser.add_argument('-n', '--early_stop_patience', default=None, type=int)

            # add_ReduceLROnPlateau
            parser.add_argument('-rlr', '--add_ReduceLROnPlateau', action='store_true')

            args = parser.parse_args(sys.argv[2:len(sys.argv)])

            input_file = args.input
            test_file = args.test
            out_dir = args.out_dir
            max_rt = args.max_rt
            max_length = args.max_length
            mod = args.mod
            unit = args.unit

            use_radam = args.radam

            if mod is not None:
                mod = mod.split(",")

            epochs = args.epochs
            batch_size = args.batch_size

            model_file = args.model_file
            ga = args.ga

            early_stop_patience= args.early_stop_patience
            add_reverse = args.add_reverse
            add_ReduceLROnPlateau = args.add_ReduceLROnPlateau

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            scale_para = dict()
            scale_para['scaling_method'] = args.scale_method
            scale_para['rt_max'] = max_rt

            scale_factor = args.scale_factor
            if scale_factor is not None:
                scale_para['scaling_factor'] = scale_factor

            print("Scaling method: %s" % (str(scale_para['scaling_method'])))

            ensemble_models(input_data=input_file, test_file=test_file, nb_epoch=epochs, batch_size=batch_size,
                            scale_para=scale_para, max_x_length=max_length, mod=mod, unit=unit, models_file=model_file,
                            ga_file=ga,
                            out_dir=out_dir, use_radam=use_radam, early_stop_patience=early_stop_patience,
                            add_reverse=add_reverse,add_ReduceLROnPlateau=add_ReduceLROnPlateau)

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






