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


def validate(model, loader, device):
    # set model to evaluation mode
    model.eval()

    outputs = list()
    targets = list()

    with torch.no_grad():
        for i, (batch, target) in enumerate(loader):
            # move batch and target to GPU on same device
            # as model and criterion
            batch.to(device=device)
            target.to(device=device)

            targets.append(target)

            # perform forward pass and update n_correct
            with torch.no_grad():
                preds = model(batch)
                outputs.append(preds)

    return outputs, targets
