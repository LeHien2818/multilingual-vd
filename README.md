# MoEVulD - MoE Vulnerability Detection

## Prerequisites
- Anaconda
- Python 3.10
- CUDA (Recommended)

## Setup
### Step 1: Clone this repository: 
```
https://github.com/LeHien2818/multilingual-vd.git
```

### Step 2: Install required dependecies:
### Windows
```
conda env create -f src/moe/env_window.yml
```
### Linux
```
conda env create -f src/moe/environment.yml
```

## Usage
### Baseline and Baseline w MulVuln
For Linux: Allowing two follow files can be executed:
```
chmod +x src/codebert/train.sh
chmod +x src/codebert/inference.sh
```
Modify `src/codebert/config.py` to decide data in which language going to be used:
```
LANGUAGE = "CCPP" //  c/c++
LANGUAGE = "Python" // python
LANGUAGE = "" // all data
``` 
#### Training:
For training phase, modify the `train.sh` file to start training. Example file:
```
python <filename> \
    --output_dir=<path_to_the_model_dir> \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --train_data_file=<path_to_train_data_file> \
    --eval_data_file=<path_to_validation_data_file> \
    --test_data_file=<path_to_test_data_file> \
    --num_train_epochs <train epochs number> \
    --block_size 256 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 45  2>&1 | tee <log_file_name>.log
```
Set filename to `run_debug.py` if run the baseline classification and set filename to `run_debug_mulvul.py` if run the baseline with mulvul technique.

#### Inference:
For inference phase, modify the `inference.sh` file to start training. Example file:
```
python <file_name> \
    --output_dir=<path_to_model_dir> \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_test \
    --train_data_file=<path_to_train_data_file> \
    --eval_data_file=<path_to_validation_data_file> \
    --test_data_file=<path_to_test_data_file> \
    --num_train_epochs 1 \
    --block_size 64 \
    --train_batch_size 8 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --seed 123456  2>&1 | tee test_sven_cpp_1_baseline.log
```

Set filename to `run_debug.py` if run the baseline classification and set filename to `run_debug_mulvul.py` if run the baseline with mulvul technique.

### MoEVulD

Modify the `src/moe/config_moe.py` on purpose for running:
```
BATCH_SIZE = 16
EPOCHS = <number of training epochs>
F1_BEST_STATE = True // Set to False if want the best Accuracy

MODEL_SAVE_DIR = "<saved_model_dir>"
TEST_MODE = False  // Set to True if just need for inference

TRAIN_DATA_PATH = "<path_to_train_data_file>"
VAL_DATA_PATH = "<path_to_validation_data_file>"
TEST_DATA_PATH = "<path_to_test_data_file>"
```

#### MoEVulD without MulVulAssistant
```
python MoE_mulvuln.py 
```
#### MoEVulD with MulVulAssistant
```
python MoE.py 
```

## Dataset
PrimeVul and SVEN is 2 datasets used in this thesis. Link to download the dataset: https://drive.google.com/drive/folders/1puwq6FymUgpjrOBPGzO2BOoqSlqKLZE6?usp=sharing
### Data description
```
- code : the source code
- vuln : vuln = 1 means code is vulnerable, vul = 0 means code is benign
- language : programming language of the source code
- cwe: cwe of the source code, cwe = -1 means No-CWE, cwe = -2 means Other-CWE
```