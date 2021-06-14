Comparison / interaction study on Batch/Group/Layer Normalizations and L1/L2 losses

# Group: EVA6 - Group 3
1. Muhsin Abdul Mohammed - muhsin@omnilytics.co 
2. Nilanjana Dev Nath - nilanjana.dvnath@gmail.com
3. Pramod Ramachandra Bhagwat - pramod@mistralsolutions.com
4. Udaya Kumar NAndhanuru - udaya.k@mistralsolutions.com
------

  * [Notebook](#notebook)
  * [File Structure](#file-structure)
  * [Model](#model)
    + [Parametrized Model Class](#parametrized-model-class)
    + [Model Architecture](#model-architecture)
  * [Graphs](#graphs)
  * [Training Log](#training-log)
  * [Misclassified](#misclassified)
  * [normalization sheet](#normalization-sheet)
  * [Analysis](#analysis)

## Notebook
View Notebook -- [training_assignment_v3.ipynb](https://nbviewer.jupyter.org/github/askmuhsin/cnn_regularizations/blob/e7b2823f678b692cd7590c8c2e0c2415d7fbff97/training_nbs/training_assignment_v3.ipynb) <br>
Run Notebook [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/askmuhsin/cnn_regularizations/HEAD?filepath=%2Ftraining_nbs%2Ftraining_assignment_v3.ipynb)

## File Structure
```
└── utils
    ├── __init__.py
    ├── data.py                 ## dataloaders, augmentations, transforms
    ├── misc.py                 ## helpers, loggers, analysis
    ├── model.py                ## nn model
    ├── regularizations.py      ## not used
    ├── testing.py              ## testing flow
    └── training.py             ## training flow
```

## Model
### Parametrized Model Class
Regularizers can be optionally selected during model instansiation.<br>
User can instantiate models with/without dropout, BN, GN, LN by -- 
```python
from utils.model import Net

net = Net(dropout_value=0.05, BN=True).to(device)                   ## dropout + BN
net = Net(dropout_value=0, BN=False, LN=True).to(device)            ## only LN
net = Net(dropout_value=0, BN=True, LN=True, GN=True).to(device)    ## BN + LN + GN (default group is size 2)
```
_Note : by default droput with 0.05 and BN is selected, so have to set it to manually off if required_
<br><br>
The block that does this is in `utils.model` -
```python
    def build_conv_block(
        self,
        in_channel, out_channel,
        kernel_size=(3, 3),
        padding=0,
    ):
        elements = []
        conv_layer = nn.Conv2d(
            in_channels=in_channel, 
            out_channels=out_channel, 
            kernel_size=kernel_size, 
            padding=padding, 
            bias=False
        )
        activation_layer = nn.ReLU()
        elements.extend([conv_layer, activation_layer])
        
        regularizers = []
        if self.dropout_value:
            regularizers.append(nn.Dropout(self.dropout_value))
        if self.BN:
            regularizers.append(nn.BatchNorm2d(out_channel))
        if self.LN:
            regularizers.append(nn.GroupNorm(1, out_channel))
        if self.GN:
            regularizers.append(nn.GroupNorm(self.GN_groups, out_channel))
        elements.extend(regularizers)
        
        return nn.Sequential(*elements)
 ```
 
### Model Architecture
- *Network with Group Normalization + L1*
<img src="https://github.com/askmuhsin/cnn_regularizations/blob/main/resources/GN__L1_model.png" alt="GN_L1" width="300"/>

- *Network with Layer Normalization + L2*
<img src="https://github.com/askmuhsin/cnn_regularizations/blob/main/resources/LN__L2_model.png" alt="LN_L2" width="300"/>

- *Network with L1 + L2 + BN* 
<img src="https://github.com/askmuhsin/cnn_regularizations/blob/main/resources/BN__L1__L2_model.png" alt="BN__L1__L2_log" width="300"/>
 
## Graphs
<img src="https://github.com/askmuhsin/cnn_regularizations/blob/main/resources/acc_charts.png" alt="acc_charts" width="400"/><img src="https://github.com/askmuhsin/cnn_regularizations/blob/main/resources/loss_charts.png" alt="loss_charts" width="400"/>

## Training Log
- *Network with Group Normalization + L1*
<img src="https://github.com/askmuhsin/cnn_regularizations/blob/main/resources/GN__L1_log.png" alt="GN_L1" width="300"/>

- *Network with Layer Normalization + L2*
<img src="https://github.com/askmuhsin/cnn_regularizations/blob/main/resources/LN__L2_log.png" alt="LN_L2" width="300"/>

- *Network with L1 + L2 + BN* 
<img src="https://github.com/askmuhsin/cnn_regularizations/blob/main/resources/BN__L1__L2_log.png" alt="BN__L1__L2_log" width="300"/>


## Misclassified
- *Misclassified -- Group Normalization + L1*
<img src="https://github.com/askmuhsin/cnn_regularizations/blob/main/resources/GN__L1_wrong_preds.png" alt="GN__L1_wrong_preds" width="500"/>

- *Misclassified -- Layer Normalization + L2*
<img src="https://github.com/askmuhsin/cnn_regularizations/blob/main/resources/LN__L2_wrong_preds.png" alt="LN__L2_wrong_preds" width="500"/>

- *Misclassified -- L1 + L2 + BN* 
<img src="https://github.com/askmuhsin/cnn_regularizations/blob/main/resources/BN__L1__L2_wrong_preds.png" alt="BN__L1__L2_wrong_preds" width="500"/>


## normalization sheet
[LINK TO NORMALIZATION SHEET](https://docs.google.com/spreadsheets/d/1Z6VSqc_8zF97ixlkd4tSgOOCbAmFte5OcpZuBNg2jwo/edit?usp=sharing)<br>
<img src="https://github.com/askmuhsin/cnn_regularizations/blob/main/resources/Normalization_screenshot.png" alt="Normalization_screenshot" width="1000"/>

## Analysis
- The previous version of the model only had BatchNormalization and Dropout as Regularizations (other than image augmentations). 
- In the current version we additionally tried L1 (lasso regularization), L2 (ridge regularization), Layer Normalization and Group Normalization.
- Generally addition of none of the above regularizations performed any better than BN alone. (Atleast not within the first 25 epcohs).
- L1 particularly increased the loss everytime. This is to be expected. l1_lambda was set to a small value. 
- L1 regularized models also had low test accuracy.
- BN, L1, and L2 combo had the least accuracy in the first few epochs, but performance increase consistently.
- LN + L2 was the best out of the 3 current experiments. But it was also underperforming compared to the previous version of the model which just had BN.


