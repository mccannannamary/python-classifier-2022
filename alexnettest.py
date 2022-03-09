import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets

class AlexMod(nn.Module):
    def __init__(self, n_hidden_units):
        super().__init__()

        # feature extractor part of model
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 3))
        self.act1 = nn.ELU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)



        # classifier part of model

