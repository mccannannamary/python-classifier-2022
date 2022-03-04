#!/usr/bin/env python

# Import libraries
import os
import pickle
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def evaluate_pretrain_model(input_dir, model_dir):

    train_dir = os.path.join(input_dir, 'train')
    val_dir = os.path.join(input_dir, 'val')
    idx_dir = input_dir

    fname = os.path.join(idx_dir, 'idx_train.npy')
    idx_train = np.load(fname)

    fname = os.path.join(idx_dir, 'idx_val.npy')
    idx_val = np.load(fname)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
    ])

    train_set = datasets.ImageFolder(root=train_dir, transform=transform)
    valid_set = datasets.ImageFolder(root=val_dir, transform=transform)

    fname = os.path.join(model_dir,'alexnet1.pkl')
    with open(fname, 'rb') as f:
        model = pickle.load(f)

    # train stats
    preds = model.predict(train_set)
    targets = np.array(train_set.targets)
    acc_t, sens_t, spec_t = get_stats(preds, targets)

    # val stats
    preds = model.predict(valid_set)
    pred_proba = model.predict_proba(valid_set)
    targets = np.array(valid_set.targets)
    acc_v, sens_v, spec_v = get_stats(preds, targets)

    # # patient val stats
    # n_classes = 2
    # val_pts = np.unique(idx_val)
    # for pt in val_pts:
    #     idx = np.where(pt == idx_val)[0]
    #     targets = np.array(valid_set.targets)[idx]
    #     pt_pred_proba = pred_proba[idx]
    #     pt_pred = preds[idx]

    print(f'Train stats:\n   Accuracy: {acc_t:.4f}, Sensitivity: {sens_t:.4f}, Specificity: {spec_t:.4f}')
    print(f'Validation stats:\n   Accuracy: {acc_v:.4f}, Sensitivity: {sens_v:.4f}, Specificity: {spec_v:.4f}')


def get_stats(preds, targets):
    # accuracy
    n_correct = (preds == targets).sum()
    acc = float(n_correct) / float(len(targets))

    # sensitivity - true positives (abnormals)
    idx = np.where(targets == 0)[0]
    n_correct = (preds[idx] == targets[idx]).sum()
    sens = float(n_correct) / float(len(idx))

    # specificity - true negatives (normals)
    idx = np.where(targets == 1)[0]
    n_correct = (preds[idx] == targets[idx]).sum()
    spec = float(n_correct) / float(len(idx))

    return acc, sens, spec






