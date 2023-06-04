#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import torch
import argparse
import sys
import os
import time
import torch.optim as optim
from torchvision import datasets, transforms
from collections import OrderedDict
from sampling import pacs_iid, pacs_noniid
##
from sag_resnet import sag_resnet
from loss_sag import *
from utils_sag import *

def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    args.dataset == 'pacs'
    all_domains = ['art_painting', 'cartoon', 'sketch', 'photo']
    if args.sources[0] == 'Rest': #(3,1)
        args.sources = [d for d in all_domains if d not in args.targets]
    if args.targets[0] == 'Rest': #(1,3)
        args.targets = [d for d in all_domains if d not in args.sources]

    #Set dataset :: Sagnet's Train_sag's Init Loader
    train_dataset, val_dataset, test_dataset,trClssnum = init_loader(args)

    # sample training data amongst users
    if args.iid:
        # Sample IID user data from Mnist
        user_groups = pacs_iid(train_dataset,trClssnum, args.num_users ,domain_set=1)
    else:
        # Sample Non-IID user data from Mnist
        user_groups = pacs_noniid(train_dataset, args.num_users)

    return train_dataset, val_dataset,test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return


##Sagnet
def init_loader(args):
    global loader_srcs, loader_vals, loader_tgts
    global num_classes

    # Set transforms
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    trans_list = []
    trans_list.append(transforms.RandomResizedCrop(args.crop_size, scale=(0.5, 1)))
    if args.colorjitter:
        trans_list.append(transforms.ColorJitter(*[args.colorjitter] * 4))
    trans_list.append(transforms.RandomHorizontalFlip())
    trans_list.append(transforms.ToTensor())
    trans_list.append(transforms.Normalize(*stats))

    train_transform = transforms.Compose(trans_list)
    test_transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats)])

    # Set datasets

    from data.pacs import PACS
    image_dir = os.path.join(args.dataset_dir, args.dataset, 'images', 'kfold')
    split_dir = os.path.join(args.dataset_dir, args.dataset, 'splits')

    print('--- Training ---')
    dataset_srcs = [PACS(image_dir,
                         split_dir,
                         domain=domain,
                         split='train',
                         transform=train_transform)
                    for domain in args.sources]

    trClssnum=[PACS(image_dir,
                     split_dir,
                     domain=domain,
                     split='train',
                     transform=train_transform).__len__() for domain in args.sources]

    print('--- Validation ---')
    dataset_vals = [PACS(image_dir,
                         split_dir,
                         domain=domain,
                         split='crossval',
                         transform=test_transform)
                    for domain in args.sources]
    print('--- Test ---')
    dataset_tgts = [PACS(image_dir,
                         split_dir,
                         domain=domain,
                         split='test',
                         transform=test_transform)
                    for domain in args.targets]
    num_classes = 7

    return dataset_srcs, dataset_vals, dataset_tgts,trClssnum