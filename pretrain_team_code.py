#!/usr/bin/env python

# Import libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from skorch.callbacks import Checkpoint
from pretrain_model_utils import PretrainedModel


def pretrain_challenge_model(input_folder):
    train_dir = os.path.join(input_folder, 'train')
    val_dir = os.path.join(input_folder, 'val')
    batch_size=10
    classes = ("abnormal", "normal")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
    ])

    train_set = datasets.ImageFolder(root=train_dir, transform=transform)
    valid_set = datasets.ImageFolder(root=val_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=batch_size)

    # Create callback, which is a learning rate scheduler that uses
    # torch.optim.lr_scheduler.StepLR to scale learning rates by
    # gamma=0.1 every 7 steps
    lrscheduler = LRScheduler(policy='StepLR', step_size=7, gamma=0.1)

    # Create a checkpoint callback, which saves the best model by
    # monitoring validation accuracy
    checkpoint = Checkpoint(f_params='best_model.pt', monitor='valid_acc_best')

    # Create a torch.device() which should be the GPU if CUDA is available,
    # otherwise use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNetClassifier(
        module=PretrainedModel,
        criterion=nn.CrossEntropyLoss,
        lr=0.001,
        batch_size=batch_size,
        max_epochs=100,
        module_output_features=2,
        optimizer=optim.SGD,
        optimizer__momentum=0.9,
        iterator_train__shuffle=True,
        iterator_train__num_workers=4,
        iterator_valid__shuffle=True,
        iterator_valid__num_workers=4,
        train_split=predefined_split(valid_set),
        callbacks=[lrscheduler, checkpoint],
        device=device
    )

    model.fit(train_set, y=None)
