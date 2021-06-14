Comparison / interaction study on Batch/Group/Layer Normalizations and L1/L2 losses

# Group: EVA6 - Group 3
1. Muhsin Abdul Mohammed - muhsin@omnilytics.co 
2. Nilanjana Dev Nath - nilanjana.dvnath@gmail.com
3. Pramod Ramachandra Bhagwat - pramod@mistralsolutions.com
4. Udaya Kumar NAndhanuru - udaya.k@mistralsolutions.com
------

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
## Parametrized Model Class
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
