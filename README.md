# [AutoRT](https://doi.org/10.1038/s41467-020-15456-w)
Peptide retention time prediction using deep learning


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
AutoRT is a python3 package. [TensorFlow](https://www.tensorflow.org/) (>=**2.5**) is supported. Its dependencies can be installed via
```shell
$ pip install -r requirements.txt
```

AutoRT has been tested on both Linux and Windows systems. It supports training and prediction on both CPU and GPU, but GPU is recommended for model training.

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
usage: autort.py [-h] -i INPUT [-o OUT_DIR] [-e EPOCHS] [-b BATCH_SIZE]
                 [-r2 MAX_RT] [-l MAX_LENGTH] [-p MOD] [-u UNIT]
                 [-sm SCALE_METHOD] [-sf SCALE_FACTOR] [-m MODEL_FILE] [-g GA]
                 [-r] [-n EARLY_STOP_PATIENCE] [-rlr]

AutoRT

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input data for training
  -o OUT_DIR, --out_dir OUT_DIR
                        Output directory
  -e EPOCHS, --epochs EPOCHS
                        The number of epochs, default is 20.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training, default is 128.
  -r2 MAX_RT, --max_rt MAX_RT
                        The maximum retention time. If the value is 0 (default), the maximum retention time 
                        will be automatically infered from the input training data.
  -l MAX_LENGTH, --max_length MAX_LENGTH
                        The length of the longest peptide to consider for modeling. 
                        If the value is 0 (default), it will be automatically infered from the input training data.
  -p MOD, --mod MOD     The integer number(s) used to represent modified amino acid(s) in training data. 
                        For example, if use 1 to represent M with oxidation and 
                        use 2 to represent S with phosphorylation, then the setting is '-p 1,2'
  -u UNIT, --unit UNIT  The unit of retention time in training data, s: second (default), m: minute.
  -sm SCALE_METHOD, --scale_method SCALE_METHOD
                        Scaling method for RT tranformation: min_max (default), mean_std and single_factor. 
                        This is used in training. Default is 'min_max'. The default method works well in most of cases. 
                        This should not be changed unless users know well about the meaning of these methods.
  -sf SCALE_FACTOR, --scale_factor SCALE_FACTOR
                        This is only useful when 'single_factor' is set for '-sm'.
  -m MODEL_FILE, --model_file MODEL_FILE
                        Trained model file. Only useful when perform transfer learning and RT prediction.
  -g GA, --ga GA        Model configuration file. Only useful when train models from scratch.
  -r, --add_reverse     Add reversed peptide in peptide encoding. This parameter will be removed in a future version.
  -n EARLY_STOP_PATIENCE, --early_stop_patience EARLY_STOP_PATIENCE
                        Number of epochs with no improvement after which training will be stopped.
  -rlr, --add_ReduceLROnPlateau
                        Reduce learning rate when a metric has stopped improving.
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

Then the setting for parameter `-p` looks like this `-p 1,2,3,4`.

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

##### An example to show how to train models from scratch:

```shell
cd example
## training
python ../autort.py train -e 100 -b 64 -g ../models/base_models/model.json -u m -p 1 -i data/PXD006109_Cerebellum_rt_add_mox_all_rt_range_3_train.tsv -sm min_max -l 48 -rlr -n 20 -o PXD006109_models/
```
After the training is finished, the trained model files are saved in the folder `PXD006109_models/`.

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

##### An example to show how to perform transfer learning:

Please note that the length of the longest peptide in the training data must be **<= 48** which is the maximum length supported by the based models in the github folder **models/base_models_PXD006109/**. 

```
$ cd example
$ cat transfer_learning.sh
## training
python ../autort.py train -i data/28CPTAC_COprospective_W_VU_20150810_05CO037_f01_normal_train.tsv -o tf_model/ -e 40 -b 64 -u m -m ../models/base_models_PXD006109/model.json -rlr -n 10

## prediction
python ../autort.py predict -t data/28CPTAC_COprospective_W_VU_20150810_05CO037_f01_normal_test.tsv -s tf_model/model.json -o tf_prediction/ -p test
```

The training took less than 20 minutes using one Titan Xp GPU on a Linux server.

A way to support prediction for peptides with length > 48 is to retrain the base models by increasing the number given to parameter "-l". For example, set "-l 60" to support peptides with length <= 60.

```
python ../autort.py train -e 100 -b 64 -g ../models/base_models/model.json -u m -p 1 -i data/PXD006109_Cerebellum_rt_add_mox_all_rt_range_3_train.tsv -sm min_max -l 60 -rlr -n 20 -o PXD006109_models/
```

The parameter setting for -e, -b, -sm, -rlr and -n for the above examples worked well for many data.

## How to cite:

Wen, B., Li, K., Zhang, Y. et al. Cancer neoantigen prioritization through sensitive and reliable proteogenomics analysis. Nature Communications 11, 1759 (2020). https://doi.org/10.1038/s41467-020-15456-w


