import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    def __init__(
        self,
        dropout_value=0.05,
        BN=True,
        LN=False,
        GN=False, GN_groups=2,
    ):
        super().__init__()
        # Regularizers
        self.BN = BN
        self.LN = LN
        self.GN = GN
        self.GN_groups = GN_groups
        self.dropout_value = dropout_value
        
        self.convblock1 = self.build_conv_block(1, 6)
        self.convblock2 = self.build_conv_block(6, 12)

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=7, kernel_size=(1, 1), padding=0, bias=False),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock4 = self.build_conv_block(7, 16)
        self.convblock5 = self.build_conv_block(16, 16)
        self.convblock6 = self.build_conv_block(16, 18)
        self.convblock7 = self.build_conv_block(18, 18, padding=1)
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )
    
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

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)

        x = self.convblock3(x)
        x = self.pool1(x)

        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)

        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
