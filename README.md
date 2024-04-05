# BARec: Improving Sequential Recommendations via Bidirectional Temporal Data Augmentation with Pre-training

## Overview

<p align="center" width="90%">
    <img src="BARec.jpg" alt="The architecture of our proposed BARec framework." style="width: 50%; min-width: 100px; display: block; margin: auto;">
    <br>
    <b>Figure 1.</b> The architecture of our proposed BARec framework.
</p>

## Environment

```bash
pip install -r requirements.txt
```

## Datasets Preparation
**Benchmarks**: Amazon Review datasets Beauty, Movie Lens and Cell_Phones_and_Accessories. 
The data split is done in the `leave-one-out` setting. Make sure you download the datasets from the [link](https://jmcauley.ucsd.edu/data/amazon/). Please, use the `DataProcessing.py` under the `data/`, and make sure you change the DATASET variable value to your dataset name, then you run:

```
python DataProcessing.py
```

## Pre-training & Fine-tuning
### Amazon Beauty 
* Reversely Pre-training and Short Sequence Augmentation
    ```
    python -u main.py --dataset=Beauty \
                    --lr=0.001 --maxlen=100 --dropout_rate=0.7 --evalnegsample=100 \
                    --hidden_units=128 --num_blocks=2 --num_heads=4 \
                    --reversed=1 --reversed_gen_num=50 --M=50 \ 
                    2>&1 | tee pre_train.log
    ```
* Next-Item Prediction with Reversed-Pre-Trained Model and Augmented dataset
    ```
    python -u main.py --dataset=Beauty \
                    --lr=0.001 --maxlen=100 --dropout_rate=0.7 --evalnegsample=100 \
                    --hidden_units=128 --num_blocks=2 --num_heads=4 \
                    --reversed_pretrain=1 --aug_traindata=47 --M=50 \
                    2>&1 | tee fine_tune.log
    ```

### Amazon Cell_Phones_and_Accessories
* Reversely Pre-training and Short Sequence Augmentation
    ```
    python -u main.py --dataset=Cell_Phones_and_Accessories \
                    --lr=0.001 --maxlen=100 --dropout_rate=0.5 --evalnegsample=100 \
                    --hidden_units=32 --num_blocks=2 --num_heads=2 \
                    --reversed=1 --reversed_gen_num=50 --M=50 \ 
                    2>&1 | tee pre_train.log
    ```
* Next-Item Prediction with Reversed-Pre-Trained Model and Augmented dataset
    ```
    python -u main.py --dataset=Cell_Phones_and_Accessories \
                    --lr=0.001 --maxlen=100 --dropout_rate=0.5 --evalnegsample=100 \
                    --hidden_units=128 --num_blocks=2 --num_heads=2 \
                    --reversed_pretrain=1 --aug_traindata=47 --M=50 \
                    2>&1 | tee fine_tune.log
    ```

### Shell Command
To simplify the process, you can run the following command:

```bash
sh run_pre_training.sh 
sh run_fine_tuning.sh
```
