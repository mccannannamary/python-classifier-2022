#!/usr/bin/env python

# Import libraries
import os
import pickle
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from pretrain_model_utils import ImageFolderWithNames

def evaluate_pretrain_model(input_dir, model_dir):

    train_dir = os.path.join(input_dir, 'train')
    val_dir = os.path.join(input_dir, 'val')
    idx_dir = input_dir

    fname = os.path.join(idx_dir, 'idx_train.npy')
    idx_train = np.load(fname)

    fname = os.path.join(idx_dir, 'idx_test.npy')
    idx_val = np.load(fname)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
    ])

    train_set = datasets.ImageFolder(root=train_dir, transform=transform)
    valid_set = ImageFolderWithNames(root=val_dir, transform=transform)

    fname = os.path.join(model_dir,'alexnet_fhs.pkl')
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

    # combine predictions for FHS pictures from same patient
    pt_preds = list()
    pt_targets = list()
    val_pts = np.unique(idx_val)
    img_names = [valid_set.get_name(i) for i in range(len(valid_set))]
    for pt in val_pts:
        idx = [i for i, s in enumerate(img_names) if pt in s]
        pt_targets.append(targets[idx][0])  # all targets are same for single patient, take first one
        pt_pred_proba = pred_proba[idx]
        pt_pred = preds[idx]
        pt_preds.append(round(np.mean(pt_pred)))

    pt_preds = np.vstack(pt_preds)
    pt_targets = np.vstack(pt_targets)
    acc_p, sens_p, spec_p = get_stats(pt_preds, pt_targets)

    print(f'Train stats:\n   Accuracy: {acc_t:.4f}, Sensitivity: {sens_t:.4f}, Specificity: {spec_t:.4f}')
    print(f'Validation stats:\n   Accuracy: {acc_v:.4f}, Sensitivity: {sens_v:.4f}, Specificity: {spec_v:.4f}')
    print(f'Patient stats:\n   Accuracy: {acc_p:.4f}, Sensitivity: {sens_p:.4f}, Specificity: {spec_p:.4f}')

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






