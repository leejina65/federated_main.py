#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=100, #10->50
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.5,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=25,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=16,
                        help="local batch size: B") #FL=10 sagnet=32
    parser.add_argument('--lr', type=float, default=0.002, #0.01 ->0.001
                        help='learning rate') #sagnet=0.04
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)') #sagnet=0.9

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=3, help="number \
                        of channels of imgs") #FG=1
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # other arguments
    parser.add_argument('--num_classes', type=int, default=7, help="number \
                        of classes")
    parser.add_argument('--gpu', default=0, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type \
                        of optimizer")
    parser.add_argument('--iid', type=int, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--seed', type=int, default=1, help='random seed')


    parser.add_argument('--dataset-dir', type=str, default='C:/Users/BamiDeep1/Desktop/FLEG0420/dataset',
                        help='Sagnet:: home directory to dataset')
    parser.add_argument('--dataset', type=str, default='pacs',
                        help="Sagnet::dataset name")
    parser.add_argument('--sources', type=str, default=['Rest'],nargs='*',
                        help='Sagnet::domains for train')
    parser.add_argument('--targets', type=str, default=['cartoon'],nargs='*',
                        help='Sagnet::domains for test')
    parser.add_argument('--save-dir', type=str, default='checkpoint',
                        help='Sagnet::home directory to save model')
    parser.add_argument('--method', type=str, default='sagnet',
                        help='Sagnet::method name')
    parser.add_argument('--workers', type=int, default=4,
                        help='Sagnet::number of workers')

    parser.add_argument('--input-size', type=int, default=256,
                        help='Sagnet::input image size')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='Sagnet::crop image size')
    parser.add_argument('--colorjitter', type=float, default=0.4,
                        help='Sagnet::color jittering')

    parser.add_argument('--arch', type=str, default='sag_resnet',
                        help='Sagnet::network archiecture')
    parser.add_argument('--depth', type=str, default='18',
                        help='Sagnet::depth of network')
    parser.add_argument('--drop', type=float, default=0.5,
                        help='Sagnet::dropout ratio')

    parser.add_argument('--sagnet', action='store_true', default=True,
                        help='use sagnet')
    parser.add_argument('--style-stage', type=int, default=3,
                        help='stage to extract style features {1, 2, 3, 4}')
    parser.add_argument('--w-adv', type=float, default=0.1,
                        help='weight for adversarial loss')

    parser.add_argument('--from-sketch', action='store_true', default=False,
                        help='training from scratch')

    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--iterations', type=int, default=100, #2000
                        help='number of training iterations')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        help='learning rate scheduler {step, cosine}')
    parser.add_argument('--milestones', type=int, nargs='+', default=[1000, 1500],
                        help='milestones to decay learning rate (for step scheduler)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='gamma to decay learning rate')

    parser.add_argument('--clip-adv', type=float, default=0.1,
                        help='grad clipping for adversarial loss')

    parser.add_argument('--log-interval', type=int, default=10,
                        help='iterations for logging training status')
    parser.add_argument('--log-test-interval', type=int, default=10,
                        help='iterations for logging test status')
    # parser.add_argument('--test-interval', type=int, default=args.local_ep,
    #                     help='iterations for test')

    args = parser.parse_args()
    return args
