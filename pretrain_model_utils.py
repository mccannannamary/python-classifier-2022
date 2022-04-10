import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets


class ImageFolderWithNames(datasets.ImageFolder):
    """Custom dataset that includes image file names. Extends
    torchvision.datasets.ImageFolder"""

    # add new method that returns name of image when called
    def get_name(self, index):
        path = self.imgs[index][0]
        path = path.split('/')[3]
        name = path.split('\\')[2]

        return name


class ResNet18(nn.Module):
    def __init__(self, n_classes, pretrained_weights):
        super().__init__()

        # load a pre-trained ResNet18
        model = models.resnet18(pretrained=pretrained_weights)

        # Freeze layers in all but final linear layer
        # for param in model.parameters():
        #     param.requires_grad_(False)
        #
        # for param in model.fc.parameters():
        #     param.requires_grad_(True)

        n_in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=n_in_features, out_features=n_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)


class ResNet50(nn.Module):
    def __init__(self, n_classes, pretrained_weights):
        super().__init__()

        # load a pre-trained ResNet50
        model = models.resnet50(pretrained=pretrained_weights)

        # replace final linear layer with correct number of output features
        n_in_features = model.fc.in_features
        model.fc = nn.Linear(in_features=n_in_features, out_features=n_classes, bias=True)

        self.model = model

    def forward(self, x):
        return self.model(x)
