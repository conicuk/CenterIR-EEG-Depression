# CenterIR-EEG-Depression

This repository contains the official PyTorch implementation of the paper

**"CenterIR: An Imbalance-Aware Deep Regression Framework for EEG-Based Depression Severity Estimation in Older Adults"**

Last update : 2026/01/23


## Abstract

Electroencephalography (EEG)-based mental health assessment has gained increasing attention as a non-invasive tool for quantifying depression severity in older adults. However, regression models for continuous severity prediction remain limited, particularly under imbalanced data distributions. This study presents a deep learning framework that integrates convolutional neural networks and bidirectional long short-term memory modules with a novel CenterIR loss to enhance regression performance.

The proposed method is as follows:

1. **CNN Module:** Extracts both temporal and spatial features from EEG signals.

2. **Bi-LSTM Module:** Captures bidirectional temporal context. 

3. **CenterIR Loss Function:** Derives continuous feature embeddings from unbalanced regression learning.


## File Structure

```bash
├── README.md           # Project documentation
├── loss_CenterIR.py    # Implementation of the proposed CenterIR loss function
├── model.py            # CNN-Bi-LSTM architecture definition
├── requirements.txt    # Dependencies and version information
├── run.py              # Main entry point to run
└── train.py            # Training and validation procedures

```

## Dependencies

This project is implemented based on **PyTorch**.  
The following core dependencies are recommended to run the code properly.

> - python >= 3.10
> - torch = 2.7.0+cu118
> - scikit-learn = 1.6.1
> - numpy

All experimental dependencies and version details can be found in `requirements.txt`.


## Usage

### 1. Data Preparation

Both input data and targets are expected as NumPy (`.npy`) files.

- **Input**: NumPy array with shape `(N, 1, C, T)`
  - `N`: number of samples
  - `C`: number of channels
  - `T`: number of time points
- **Target**: NumPy array with shape `(N, 1)`

The dataset used in this study is not publicly available.
To verify that the training loop runs correctly, dummy data with the same format is provided.

Once the data is prepared, load the input and target data in `run.py` as follows:

```python
import numpy as np

input_np = np.load("path/to/input.npy")    # shape: (N, 1, C, T)
target_np = np.load("path/to/target.npy")   # shape: (N, 1)

```


### 2. Training

To train the model using 10-Fold Cross-Validation, run:

```python
python run.py

```

### 3. Hyperparameters

Key hyperparameters can be configured directly in `run.py`.:

```python
BATCH_SIZE = 32
Learning_Rate = 0.0001
EPOCHS = 100

## CenterIR hyperparameters
boundaries = 16    # Initial boundary value; final sorted boundaries can be smaller than the initial setting
k = [1, 3, 5, 15]    # The number of adjacent boundaries grouped together
CenterIR_lambda = 5e-8    # CenterIR's weight parameter

```


## Citation

If you find this work useful in your research, please consider citing our paper:

**"CenterIR: An Imbalance-Aware Deep Regression Framework for EEG-Based Depression Severity Estimation in Older Adults"**

(The paper is currently under-review.)


