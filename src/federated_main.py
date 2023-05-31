#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch.optim as optim
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from torchvision import transforms
from options import args_parser #parameters import
from update import LocalUpdate, test_inference
from utils import pacs_noniid, pacs_iid

#from models import MLP, ResNet
from collections import OrderedDict
from sag_resnet import sag_resnet
from utils import get_dataset, average_weights, exp_details
from loss_sag import *
from utils_sag import *

def init_loader(): #train_dataset, val_dataset,test_dataset):
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

    if args.dataset=='pacs':
        args.dataset == 'pacs'
        all_domains = ['art_painting', 'cartoon', 'sketch', 'photo']
        if args.sources[0] == 'Rest':  # (3,1)
            args.sources = [d for d in all_domains if d not in args.targets]
        if args.targets[0] == 'Rest':  # (1,3)
            args.targets = [d for d in all_domains if d not in args.sources]

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

        trClssnum = [PACS(image_dir,
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

        if args.iid:
            # Sample IID user data from Mnist
            user_groups = pacs_iid(dataset_srcs, trClssnum, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            user_groups = pacs_noniid(dataset_srcs, args.num_users)

        return dataset_srcs, dataset_vals, dataset_tgts, user_groups

    # # Set loaders
    # kwargs = {'num_workers': args.workers, 'pin_memory': True}
    # loader_srcs = [torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.local_bs,
    #     shuffle=True,
    #     drop_last=True,
    #     **kwargs)
    #     for dataset in dataset_srcs]
    # loader_vals = [torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=int(args.local_bs * 4),
    #     shuffle=False,
    #     drop_last=False,
    #     **kwargs)
    #     for dataset in dataset_vals]
    # loader_tgts = [torch.utils.data.DataLoader(
    #     dataset_tgt,
    #     batch_size=int(args.local_bs * 4),
    #     shuffle=False,
    #     drop_last=False,
    #     **kwargs)
    #     for dataset_tgt in dataset_tgts]

def init_optimizer(model):
    global optimizer, optimizer_style, optimizer_adv
    global scheduler, scheduler_style, scheduler_adv
    global criterion, criterion_style, criterion_adv

    # Set hyperparams
    optim_hyperparams = {'lr': args.lr,
                         'weight_decay': args.weight_decay,
                         'momentum': args.momentum}
    if args.scheduler == 'step':
        Scheduler = optim.lr_scheduler.MultiStepLR
        sch_hyperparams = {'milestones': args.milestones,
                           'gamma': args.gamma}
    elif args.scheduler == 'cosine':
        Scheduler = optim.lr_scheduler.CosineAnnealingLR
        sch_hyperparams = {'T_max': args.iterations}

    # Main learning
    params = model.parameters()
    optimizer = optim.SGD(params, **optim_hyperparams) #Sagnet에 맞춰서 hyperparameters
    scheduler = Scheduler(optimizer, **sch_hyperparams)
    criterion = torch.nn.CrossEntropyLoss()


    # Style learning
    params_style = model.style_params()
    optimizer_style = optim.SGD(params_style, **optim_hyperparams)
    scheduler_style = Scheduler(optimizer_style, **sch_hyperparams)
    criterion_style = torch.nn.CrossEntropyLoss()

    # Adversarial learning
    params_adv = model.adv_params()
    optimizer_adv = optim.SGD(params_adv, **optim_hyperparams)
    scheduler_adv = Scheduler(optimizer_adv, **sch_hyperparams)
    criterion_adv = AdvLoss()

    dic = OrderedDict([('optimizer',optimizer),('scheduler',scheduler),('criterion',criterion)
                       ,('params_style',params_style),('optimizer_style',optimizer_style),('scheduler_style',scheduler_style)
                       ,('criterion_style',criterion_style),('params_adv',params_adv),('optimizer_adv',optimizer_adv),('scheduler_adv',scheduler_adv)
                       ,('criterion_adv',criterion_adv)])
    return dic

def train_sag(model,step,data):
    global dataiter_srcs

    ## Initialize iteration
    model.train()
    epoch_loss=[] #FL

    scheduler.step()
    if args.sagnet:
        scheduler_style.step()
        scheduler_adv.step()

    ## Load data
    tic = time.time() #fd: start_time

    #split DATA,LABEL to use for training
    n_srcs = len(args.sources) #art_painting, ...
    if step == 0:
        dataiter_srcs = [None] * n_srcs
    data = [None] * n_srcs
    label = [None] * n_srcs
    for i in range(n_srcs):
        if step % len(loader_srcs[i]) == 0:
            dataiter_srcs[i] = iter(loader_srcs[i])
        data[i], label[i] = next(dataiter_srcs[i])

    data = torch.cat(data)
    label = torch.cat(label)
    rand_idx = torch.randperm(len(data))
    data = data[rand_idx]
    label = label[rand_idx].cuda()

    time_data = time.time() - tic

    ## Process batch
    tic = time.time()

    # forward
    y, y_style = model(data)

    if args.sagnet:
        # learn style
        loss_style = criterion(y_style, label) #crossentropy
        optimizer_style.zero_grad() #gradient 0:: .grad 초기화
        loss_style.backward(retain_graph=True) #backpropagation: udpate weight, 각 parameter.grad<- 변화도 저장
        optimizer_style.step() #update parameter:: 변화도 반영

        # learn style_adv
        loss_adv = args.w_adv * criterion_adv(y_style)
        optimizer_adv.zero_grad()
        loss_adv.backward(retain_graph=True)
        if args.clip_adv is not None:
            torch.nn.utils.clip_grad_norm_(model.module.adv_params(), args.clip_adv)
        optimizer_adv.step()

    # learn content
    loss = criterion(y, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #loss.item
    time_net = time.time() - tic

    ## Update status
    status['iteration'] = step + 1
    status['lr'] = optimizer.param_groups[0]['lr']
    status['src']['t_data'].update(time_data)
    status['src']['t_net'].update(time_net)
    status['src']['l_c'].update(loss.item())
    if args.sagnet:
        status['src']['l_s'].update(loss_style.item())
        status['src']['l_adv'].update(loss_adv.item())
    status['src']['acc'].update(compute_accuracy(y, label))

    ## Log result
    # if step % args.log_interval == 0:
    #     print('[{}/{} ({:.0f}%)] lr {:.5f}, {}'.format(
    #         step, args.iterations, 100. * step / args.iterations, status['lr'],
    #         ', '.join(['{} {}'.format(k, v) for k, v in status['src'].items()])))


    return model.state_dict()

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    args = args_parser()
    exp_details(args)

    if args.gpu!='None':
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if not args.gpu else 'cpu'

    # BUILD MODEL
    if args.dataset == 'pacs':
        global_model= sag_resnet(depth=int(args.depth),
                   pretrained=not args.from_sketch,
                   num_classes=args.num_classes,
                   drop=args.drop,
                   sagnet=args.sagnet,
                   style_stage=args.style_stage)
        global_model=global_model.to(device)
    else:
        exit('Error: unrecognized model')

    # Initialzie loader
    print('\nInitialize loaders...from SagNet')
    dataset_srcs, dataset_vals, dataset_tgts, user_groups = init_loader()

    # Sagnet Initialize optimizer
    print('\nInitialize optimizers... from SagNet')
    opti_dic = init_optimizer(global_model)
    global_model.train()
    print(global_model)

    #Sagnet Training Process#
    # Initialize status
    src_keys = ['t_data', 't_net', 'l_c', 'l_s', 'l_adv', 'acc']
    status = OrderedDict([
        ('iteration', 0),
        ('lr', 0),
        ('src', OrderedDict([(k, AverageMeter()) for k in src_keys])),
        ('val_acc', OrderedDict([(domain, 0) for domain in args.sources])),
        ('mean_val_acc', 0),
        ('test_acc', OrderedDict([(domain, 0) for domain in args.targets])),
        ('mean_test_acc', 0),
    ])

    # Main loop 랜덤하게 초기화한 모델을 나눠주고 각자 학습시키고.. .. ..!!
    # 초기 모델 client에게 뿌리는 부분
    # 합친 global모델의 성능과 그 전의 성능을 비교

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False) #select randomly clients
        #dataset_srcs, dataset_vals, dataset_tgts, user_groups
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=dataset_srcs,idxs=user_groups[idx],
                                      logger=logger,opti=opti_dic, status = status)
            w, loss = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=dataset_srcs,idxs=user_groups[idx],
                                      logger=logger,opti=opti_dic, status = status)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, dataset_tgts)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))



