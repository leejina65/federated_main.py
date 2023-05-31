#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils_sag import *
import options
from options import args_parser
from PIL import Image
import cv2
import time
import numpy as np
from torchvision import transforms
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = [sample for dataset in dataset for sample in dataset.samples]
        a_ind = [int(j) for i in idxs for j in (i if isinstance(i, tuple) else (i,))]
        self.idxs=a_ind
#        self.args = options.args_parser()
#        self.root = self.args.dataset_dir
#        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):  #domain 0,1,2 중 특정 지정해야 함 -> generator input을 애초에 domain 지정
        image, label = self.dataset[self.idxs[item]]
        return image, label
        # path=self.root+'/pacs/images/kfold/'+image
        # img=Image.open(path).convert('RGB')
        # img_array=np.array(img)
        # return torch.tensor(img_array), torch.tensor(label)


class LocalUpdate(object): #idxs=user_groups[idx]=clinet #clinet model training
    def __init__(self, args, dataset, idxs, logger, opti, status):
        self.args = args
        self.status = status

        self.scheduler = opti['scheduler']
        self.scheduler_style = opti['scheduler_style']
        self.scheduler_adv = opti['scheduler_adv']

        self.criterion = opti['criterion']
        self.criterion_style = opti['criterion_style']
        self.criterion_adv = opti['criterion_adv']

        self.optimizer_style = opti['optimizer_style']
        self.optimizer_adv = opti['optimizer_adv']

        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[0][:int(0.8*len(idxs[0]))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        #loader for per clinet
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=1, shuffle=False) #int(len(idxs_val)/10)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=1, shuffle=False) #int(len(idxs_test)/10)
        return trainloader, validloader, testloader

    def idx_sagnet(self, model, step, images,labels, optimizer):
        args = args_parser()
        criterion = torch.nn.CrossEntropyLoss()

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

        self.scheduler.step()

        if self.args.sagnet:
            self.scheduler_style.step()
            self.scheduler_adv.step()

        ## Load data
        tic = time.time()

        self.args = options.args_parser()
        data=images
        label=labels

        root=self.args.dataset_dir+'/pacs/images/kfold/'
        imgroot=[root+data[i] for i in range(len(data))]
        img=[Image.open(img).convert("RGB") for img in imgroot]
        data=[train_transform(img) for img in img]
        data=torch.stack(data)

        rand_idx = torch.randperm(len(data))

        data = data[rand_idx]
        label = label[rand_idx].cuda()

        time_data = time.time() - tic

        ## Process batch
        tic = time.time()

        # forward
        y, y_style = model(data)

        if self.args.sagnet:
            # learn style
            loss_style = self.criterion(y_style, label)
            self.optimizer_style.zero_grad()
            loss_style.backward(retain_graph=True)
            self.optimizer_style.step()

            # learn style_adv
            loss_adv = self.args.w_adv * self.criterion_adv(y_style)
            self.optimizer_adv.zero_grad()
            loss_adv.backward(retain_graph=True)
            if self.args.clip_adv is not None:
                torch.nn.utils.clip_grad_norm_(model.adv_params(), self.args.clip_adv)
            self.optimizer_adv.step()

        # learn content
        loss = criterion(y, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_net = time.time() - tic

        self.logger.add_scalar('loss', loss.item())
        acc=compute_accuracy(y, label)


    ## Update status
        self.status['iteration'] = step + 1
        self.status['lr'] = optimizer.param_groups[0]['lr']
        self.status['src']['t_data'].update(time_data)
        self.status['src']['t_net'].update(time_net)
        self.status['src']['l_c'].update(loss.item())
        if self.args.sagnet:
            self.status['src']['l_s'].update(loss_style.item())
            self.status['src']['l_adv'].update(loss_adv.item())
        self.status['src']['acc'].update(acc)

        ## Log result
        if step % self.args.log_interval == 0:
            print('[{}/{} ({:.0f}%)] lr {:.5f}, {}'.format(
                step, self.args.iterations, 100. * step / self.args.iterations, self.status['lr'],
                ', '.join(['{} {}'.format(k, v) for k, v in self.status['src'].items()])))
        return loss_style,loss_adv,loss,y

    def update_weights(self, model, global_round): #, step, loader_srcs):
        # Set mode to train model
        model.train()
        epoch_loss = []

        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,momentum=0.5)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                model.zero_grad()
                loss_style,loss_adv,loss,y_ = self.idx_sagnet(model, iter, images,labels, optimizer)
                batch_loss.append(loss.item())

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        print(batch_loss)

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)


        for batch_idx, (images, labels) in enumerate(self.testloader):
            batch_loss,outputs = self.idx_sagnet(model, iter, images, labels, optimizer)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 0 if not args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
