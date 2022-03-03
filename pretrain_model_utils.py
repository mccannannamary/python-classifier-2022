import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


class PretrainedModel(nn.Module):
    def __init__(self, output_features):
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
        for feature in model.features:
            feature.requires_grad_(False)

        # Modify layers in classifier part for retraining, only these layers will be updated
        # during training
        model.classifier[1] = nn.Linear(in_features=9216, out_features=128)
        model.classifier[4] = nn.Linear(in_features=128, out_features=128, bias=True)
        model.classifier[6] = nn.Linear(in_features=128, out_features=2, bias=True)
        assert model.classifier[1].out_features == 128, "Should be 128"
        assert model.classifier[4].in_features == 128, "Should be 128"
        assert model.classifier[6].in_features == 128, "Should be 128"
        assert model.classifier[6].out_features == 2, "Should be 2"

        self.model = model

    def forward(self, x):
        return self.model(x)
