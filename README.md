# [AutoRT](https://doi.org/10.1038/s41467-020-15456-w)
**AutoRT** is a peptide retention time prediction tool using deep learning. It supports peptide retention prediction for **tryptic peptides** (global proteome experiments), **MHC bound peptides** (immunopeptidomics experiment) and **PTM peptides** (such as phosphoproteomics experiment, ubiquitome or acetylome experiment).

### Table of contents:

- [Performance of AutoRT](#performance-of-autort)
- [Installation](#installation)
- [Usage](#usage)
- [Run AutoRT on Google Colab](#run-autort-on-google-colab)
- [How to cite](#how-to-cite)
- [Applications](#applications)

## Performance of AutoRT
1. The performance of models trained using more than 100,000 peptides: 
[Figure 1a](https://www.nature.com/articles/s41467-020-15456-w/figures/2); 
[Supplementary Figure 1](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-15456-w/MediaObjects/41467_2020_15456_MOESM1_ESM.pdf).
2. The performance of models trained using 700~10,000 peptides with transfer learning strategy: 
[Figure 1b](https://www.nature.com/articles/s41467-020-15456-w/figures/2); 
[Supplementary Figure 3](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-15456-w/MediaObjects/41467_2020_15456_MOESM1_ESM.pdf); 
[Supplementary Figure 8-10](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-15456-w/MediaObjects/41467_2020_15456_MOESM1_ESM.pdf); 
3. The performance of models trained using label free, iTRAQ or TMT data: 
[Figure 1a](https://www.nature.com/articles/s41467-020-15456-w/figures/2); 
[Figure 1b](https://www.nature.com/articles/s41467-020-15456-w/figures/2); 
[Supplementary Figure 3](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-15456-w/MediaObjects/41467_2020_15456_MOESM1_ESM.pdf); 
[Supplementary Figure 8-10](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-15456-w/MediaObjects/41467_2020_15456_MOESM1_ESM.pdf).
4. The performance of models trained using immunopeptidomics data: 
[Supplementary Figure 6](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-15456-w/MediaObjects/41467_2020_15456_MOESM1_ESM.pdf); 
5. The performance of models trained using public iRT data: 
[Figure 1a](https://www.nature.com/articles/s41467-020-15456-w/figures/2); 
[Supplementary Figure 1](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-15456-w/MediaObjects/41467_2020_15456_MOESM1_ESM.pdf).


## Installation

#### Download AutoRT

```shell
$ git clone https://github.com/bzhanglab/AutoRT
```

#### Installation
AutoRT is a python3 package. [TensorFlow](https://www.tensorflow.org/) (>=**2.6**) is supported. Its dependencies can be installed via
```shell
$ pip install -r requirements.txt
```

AutoRT has been tested on both Linux and Windows systems. It supports training and prediction on both CPU and GPU, but GPU is recommended for model training. Multiple GPUs are also supported.

AutoRT can also be used through docker:

```shell
docker pull proteomics/autort
```

## Usage

#### Training and prediction

AutoRT supports training models from scratch as well as transfer learning. We recommend to train models using **GPU**.

##### Training:
```
$ python autort.py train -h
optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input data for training
  -o OUT_DIR, --out_dir OUT_DIR
                        Output directory
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs, default is 40.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training, default is 64.
  -r2 MAX_RT, --max_rt MAX_RT
                        The maximum retention time. If the value is 0 (default), the maximum retention time will be automatically infered from the input training data.
  -l MAX_LENGTH, --max_length MAX_LENGTH
                        The length of the longest peptide to consider for modeling. If the value is 0 (default), it will be automatically infered from the input model file.                        
  -u UNIT, --unit UNIT  The unit of retention time in training data, s: second, m: minute (default).
  -sm SCALE_METHOD, --scale_method SCALE_METHOD
                        Scaling method for RT tranformation: min_max (default), mean_std and single_factor. This is used in training. Default is 'min_max'. The default method works well in most of cases. This should not be
                        changed unless users know well about the meaning of these methods.
  -sf SCALE_FACTOR, --scale_factor SCALE_FACTOR
                        This is only useful when 'single_factor' is set for '-sm'.
  -m MODEL_FILE, --model_file MODEL_FILE
                        Trained model file. Only useful when perform transfer learning and RT prediction.
  -r, --add_reverse     Add reversed peptide in peptide encoding. This parameter will be removed in a future version.
  -n EARLY_STOP_PATIENCE, --early_stop_patience EARLY_STOP_PATIENCE
                        Number of epochs with no improvement after which training will be stopped.
  -rlr, --add_ReduceLROnPlateau
                        Reduce learning rate when a metric has stopped improving.
  -g GPU, --gpu GPU     Set gpu IDs that can be used. For example: 1,2,3.
  -d, --do_evaluation_after_each_epoch
                        Do evaluation after each epoch during model training.
  -f OUTLIER_RATIO, --outlier_ratio OUTLIER_RATIO
                        The percentage of data (outlier) will be removed from the training data using a two-step training.
```
##### Prediction:

```shell
$ python autort.py predict -h
usage: autort.py [-h] [-t TEST] [-o OUT_DIR] [-p PREFIX] [-s ENSEMBLE] [-a]

AutoRT

optional arguments:
  -h, --help            show this help message and exit
  -t TEST, --test TEST  Input data for testing
  -o OUT_DIR, --out_dir OUT_DIR
                        Output directory
  -p PREFIX, --prefix PREFIX
  -s ENSEMBLE, --ensemble ENSEMBLE
  -a, --radam
```

An example training data is available in the `example/data` folder as shown below. The training data should have at least 2 columns `x` and `y` in a **`tab-delimited text format`**. The `x` column contains peptide sequences and `y` column contains retention time. The unit of retention time can be **minute** or **second**. If the unit of retention time is **minute**, users should set "-u m" and if the unit of retention time is **second**, users should set "-u s". 

```shell
$ head example/data/PXD006109_Cerebellum_rt_add_mox_all_rt_range_3_train.tsv
x                       y
AAAAAAAAAAAAAAAGAAGK    58.056936170212765
AAAAAAAAAK              4.394
AAAAAAAAATEQQGSNGPVK    26.975
AAAAAAAAQ1HTK           8.2107
AAAAAAAAQMHTK           15.553
AAAAAASLLANGHDLAAAMAVDK 73.801
AAAAADLANR              18.85030769230769
AAAAAGAGLK              13.3316
AAAAALSGSPPQTEK         27.125375000000002
```

For a peptide with modification(s) (modified peptide), it must be formatted as described below:

Each of the modified amino acids must be represented using a different integer number (0-9) in the peptide sequence and keep consistent in both training and testing (or prediction) data. For example, if there are four modifications in a dataset: oxidation on M, phosphorylation on S,T and Y, then we can use **1** represents **oxidation of M**, **2** represents **phosphorylation of S**, **3** represents **phosphorylation of T** and **4** represents **phosphorylation of Y**. 

Then a peptide like this:
```
AS(phosphorylation)FDFADFST(phosphorylation)PY(phosphorylation)STM(oxidation)AAK 
```
must be formatted as 
```
A2FDFADFS3P4ST1AAK
```


An example testing data is available in the data folder as shown below. The first column "x" is required. The second column "y" is optional.
```
$ head example/data/PXD006109_Cerebellum_rt_add_mox_all_rt_range_3_test.tsv
x                           y
DSQFLAPDVTSTQVNTVVSGALDR    84.633
LLDLYASGER                  51.07025
EMPQNVAK                    16.787
GSPTGSSPNNASELSLASLTEK      62.60036363636364
LGLDEYLDK                   57.118833333333335
FNGAQVNPK                   20.01075
AVTISGTPDAIIQCVK            66.29124999999999
SSPVEYEFFWGPR               81.73833333333333
AIPSYSHLR                   29.95825
```

##### An example to show how to train a highly accurate RT model using a small dataset from a global proteome experiment:

Please note that the length of the longest peptide in the training data must be **<= 60** which is the maximum length supported by the based models in the github folder **models/general_base_model/**. 

```
$ cd example
$ cat transfer_learning.sh
## training
python ../autort.py train -i data/28CPTAC_COprospective_W_VU_20150810_05CO037_f01_normal_train.tsv -o tf_model/ -e 40 -b 64 -u m -m ../models/general_base_model/model.json -rlr -n 10

## prediction
python ../autort.py predict -t data/28CPTAC_COprospective_W_VU_20150810_05CO037_f01_normal_test.tsv -s tf_model/model.json -o tf_prediction/ -p test
```

The training took less than 15 minutes using one Titan Xp GPU on a Linux server.


The parameter setting for -e, -b, -sm, -rlr and -n for the above examples worked well for many data.

##### An example to show how to train models from scratch:

```shell
cd example
## training
python ../autort.py train -e 100 -b 64 -m ../models/base_model/model.json -u m -i data/PXD006109_Cerebellum_rt_add_mox_all_rt_range_3_train.tsv -sm min_max -rlr -n 20 -o PXD006109_models/
```
After the training is finished, the trained model files are saved in the folder `PXD006109_models/`. If users want to train a model to support peptides longer than 60 (e.g., 100 amino acids), set the parameter **-l** to a number longer than 60, for example, **-l 100**.

Then we can use the trained models for prediction as shown below:
```
## prediction:
python ../autort.py predict -t data/PXD006109_Cerebellum_rt_add_mox_all_rt_range_3_test.tsv -s PXD006109_models/model.json -o PXD006109_prediction/ -p test
```
The prediction result will be saved in file `PXD006109_prediction/test.csv` and looks like below. The column `y_pred` contains the predicted retention time.
```
$ head PXD006109_prediction/test.csv
x                         y	                y_pred
DSQFLAPDVTSTQVNTVVSGALDR  84.633	        87.11121
LLDLYASGER                51.07025	        51.25605
EMPQNVAK                  16.787	        17.024113
GSPTGSSPNNASELSLASLTEK    62.60036363636364	61.83924
LGLDEYLDK                 57.11883333333334	57.66608
FNGAQVNPK                 20.01075	        19.809322
AVTISGTPDAIIQCVK          66.29124999999999	68.5102
SSPVEYEFFWGPR             81.73833333333333	82.20954
AIPSYSHLR                 29.95825	        31.344095
```

The training took less than 12 hours using one Titan Xp GPU on a Linux server.

## Run AutoRT on Google Colab:

Example: [example/Experiment_specific_RT_prediction_using_AutoRT.ipynb](https://github.com/bzhanglab/AutoRT/blob/master/example/Experiment_specific_RT_prediction_using_AutoRT.ipynb)

Example: [example/Phosphorylation_experiment_specific_RT_prediction_using_AutoRT_Colab.ipynb](https://github.com/bzhanglab/AutoRT/blob/master/example/Phosphorylation_experiment_specific_RT_prediction_using_AutoRT_Colab.ipynb)


## How to cite:

Wen, B., Li, K., Zhang, Y. et al. Cancer neoantigen prioritization through sensitive and reliable proteogenomics analysis. Nature Communications 11, 1759 (2020). https://doi.org/10.1038/s41467-020-15456-w


## Applications:

A list of applications of AutoRT could be found at: [AutoRT applications](https://github.com/bzhanglab/deep_learning_in_proteomics#peptide-retention-time-prediction).
