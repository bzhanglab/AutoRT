# AutoRT
Peptide retention time prediction using deep learning

## Installation

#### Download AutoRT

```shell
$ git clone https://github.com/bzhanglab/AutoRT
```

#### Installation
AutoRT is a python3 package and its dependencies can be installed via
```shell
$ pip install -r requirements.txt
```

AutoRT has been tested on both Linux and Windows systems.

AutoRT is also available as a **docker**. AutoRT docker requires both [Docker](https://www.docker.com/) (>=19.03) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) (>=2.2.2).

```shell
docker pull proteomics/autort
```

## Usage

#### Training and prediction

AutoRT supports training models from scratch as well as transfer learning. We recommend to train models using **GPU**.

##### Training:
```
$ python autort.py train -h
usage: autort.py [-h] -i INPUT [-t TEST] [-o OUT_DIR] [-e EPOCHS]
                 [-b BATCH_SIZE] [-r2 MAX_RT] [-l MAX_LENGTH] [-p MOD]
                 [-u UNIT] [-sm SCALE_METHOD] [-sf SCALE_FACTOR]
                 [-m MODEL_FILE] [-g GA] [-a] [-r] [-n EARLY_STOP_PATIENCE]
                 [-rlr]

AutoRT

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input data for training
  -t TEST, --test TEST  Input data for testing
  -o OUT_DIR, --out_dir OUT_DIR
                        Output directory
  -e EPOCHS, --epochs EPOCHS
  -b BATCH_SIZE, --batch_size BATCH_SIZE
  -r2 MAX_RT, --max_rt MAX_RT
  -l MAX_LENGTH, --max_length MAX_LENGTH
  -p MOD, --mod MOD
  -u UNIT, --unit UNIT
  -sm SCALE_METHOD, --scale_method SCALE_METHOD
  -sf SCALE_FACTOR, --scale_factor SCALE_FACTOR
  -m MODEL_FILE, --model_file MODEL_FILE
  -g GA, --ga GA
  -a, --radam
  -r, --add_reverse
  -n EARLY_STOP_PATIENCE, --early_stop_patience EARLY_STOP_PATIENCE
  -rlr, --add_ReduceLROnPlateau
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

```
$ cd example
$ cat transfer_learning.sh
## training
python ../autort.py train -i data/28CPTAC_COprospective_W_VU_20150810_05CO037_f01_normal_train.tsv -o tf_model/ -e 40 -b 64 -u m -m ../models/base_models_PXD006109/model.json -rlr -n 10

## prediction
python ../autort.py predict -t data/28CPTAC_COprospective_W_VU_20150810_05CO037_f01_normal_test.tsv -s tf_model/model.json -o tf_prediction/ -p test
```

The training took less than 20 minutes using one Titan Xp GPU on a Linux server.

## How to cite:

Wen, B., Li, K., Zhang, Y. et al. Cancer neoantigen prioritization through sensitive and reliable proteogenomics analysis. Nature Communications 11, 1759 (2020). https://doi.org/10.1038/s41467-020-15456-w


