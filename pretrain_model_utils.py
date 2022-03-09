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


class PretrainedModel(nn.Module):
    def __init__(self, n_hidden_units):
        super().__init__()

        # load a pretrained AlexNet
        model = models.alexnet(pretrained=True)
        assert isinstance(model, models.AlexNet)

        # AlexNet consist of two parts, (1) a feature extractor, consisting of convolutional
        # layers, max pooling, and non-linearities. (2) a classifier, which consists of fully
        # connected layers (Linear) and non-linearities. The idea will be to freeze the
        # features part, and create smaller linear layers in the classifier part with less
        # hidden units than in the original model

        # Freeze layers of feature extractor part of network:
        # for feature in model.features:
        #     feature.requires_grad_(False)

        # Modify layers in classifier part for retraining, only these layers will be updated
        # during training
        model.classifier[1] = nn.Linear(in_features=9216, out_features=n_hidden_units)
        model.classifier[4] = nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units, bias=True)
        model.classifier[6] = nn.Linear(in_features=n_hidden_units, out_features=2, bias=True)
        assert model.classifier[1].out_features == n_hidden_units, "Should be 128"
        assert model.classifier[4].in_features == n_hidden_units, "Should be 128"
        assert model.classifier[6].in_features == n_hidden_units, "Should be 128"
        assert model.classifier[6].out_features == 2, "Should be 2"

        self.model = model

    def forward(self, x):
        return self.model(x)


class AlexNetMod(nn.Module):
    def __init__(self, n_hidden_units, n_classes):
        super().__init__()

        ## feature extractor part of model
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 3))
        # torch.nn.init.kaiming_normal_(self.conv1.weight)
        # torch.nn.BatchNorm2d(num_features=64)
        self.act1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        # torch.nn.init.kaiming_normal_(self.conv2.weight)
        # torch.nn.BatchNorm2d(num_features=192)
        self.act2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # torch.nn.init.kaiming_normal_(self.conv3.weight)
        # torch.nn.BatchNorm2d(num_features=384)
        self.act3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # torch.nn.init.kaiming_normal_(self.conv4.weight)
        # torch.nn.BatchNorm2d(num_features=256)
        self.act4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # torch.nn.init.kaiming_normal_(self.conv5.weight)
        # torch.nn.BatchNorm2d(num_features=256)
        self.act5 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)

        ## classifier part of model
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        self.fc1 = nn.Linear(in_features=9216, out_features=n_hidden_units, bias=True)
        self.act6 = nn.ReLU(inplace=True)

        self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        self.fc2 = nn.Linear(in_features=n_hidden_units, out_features=n_hidden_units, bias=True)
        self.act7 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(in_features=n_hidden_units, out_features=n_classes, bias=True)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.act3(self.conv3(x))
        x = self.act4(self.conv4(x))
        x = self.pool3(self.act5(self.conv5(x)))

        x = self.act6(self.fc1(self.dropout1(x.view(-1, 9216))))
        x = self.act7(self.fc2(self.dropout2(x)))

        out = self.fc3(x)

        return out
