#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import collections

'''
def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
'''#trainclass_num 350


def pacs_iid(dataset,trainclass_num, num_users, domain_set):
    domain_num=len(trainclass_num) # trainclassnum=[N,N2,N3], class_num=3
    #tempdomain for each class ==> split each class into same ratio
    #0~N-1, N~(N)+N2-1, N+N2~(N+N2)+N3-1
    #domain # == 4 (default)

    dict_users={}

    '''
    ########## equal 하게 나눌때 #####################
    num_items_list = [int(len(dataset[i].samples) / num_users) for i in range(domain_num)]
    num_items_min=min(num_items_list)
    print(num_items_min)
    '''
    for domain_idx in range(domain_num):
        data=dataset[domain_idx-1].samples
        str=len(dataset[domain_idx-2].samples) if domain_idx!=0 else 0
        end=str+len(data)-1
        ###########################################
        num_items = int(len(data) / num_users)  # client당 가지는 해당 domain data#
        ###########################################
        #num_items=num_items_min            #equal
        all_idxs = [i for i in range(str,end+1)] #0~N-1, N~(N)+N2-1, N+N2~(N+N2)+N3-1

        for i in range(num_users):  # users에게 data 나눠주기
            user_idxs = set(np.random.choice(all_idxs, num_items, replace=False))
            dict_users.setdefault(i, set()).add(tuple(user_idxs))
            all_idxs = list(set(all_idxs).difference(user_idxs))
    return dict_users #key:client, values: their data


def pacs_noniid(dataset,class_num,num_users):
    #client가 가지는 domain의 개수를 변수화한다는 의미:: 모든 client가 동일한 개수의 domain
    #
    # dataset_total = [sample for dataset in dataset for sample in dataset.samples]
    # mindomain=3 #minimun domain type
    # dict_users={}
    # num_items=0
    # for i in dataset:
    #     num_items+=int(len(i.samples)/num_users)
    # # len([sample for dataset in dataset for sample in dataset.sample])
    # # num_items=int(len(dataset_total)/num_users)
    # all_idxs=list(range(len(dataset_total)))
    #
    # for i in range(num_users):
    #     user_idxs=set(np.random.choice(all_idxs,num_items,replace=False))
    #     dict_users.setdefault(i,set()).add(tuple(user_idxs))
    #     all_idxs=list(set(all_idxs).difference(user_idxs))
    dict_users = {}
    class_num = len(class_num)
    all_idxs , num_items = [None] * class_num , [None] * class_num
    choice_domain_list = []
    for i in range(num_users):  # 유저의 개수 만큼 랜덤으로 4개의 도메인 중 2개를 뽑은 후 리스트로 저장
        choice_domain = np.random.choice (range(class_num), 2, replace=False)
        choice_domain_list.append(choice_domain) # Non-iid

        # IID code
        #choice_domain_list = np.random.randint(0, class_num, size=(num_users, 2))

    # 각 도메인마다 선택한 client의 개수를 센다
    choice_domain_num = collections.Counter(np.concatenate(choice_domain_list).tolist())

    for idx in range(class_num):
        data = dataset[idx - 1].samples
        str = len(dataset[idx - 2].samples) if idx != 0 else 0
        end = str + len(data) - 1
        domain = data

        # 각 도메인 마다 데이터와 각 client가 가져갈 개수 저장
        num_items[idx] = int(len(domain) / choice_domain_num[idx])  # client당 가지는 해당 domain data#
        all_idxs[idx] = [i for i in range(str, end + 1)]

    for i in range(num_users):  # users에게 data 나눠주기
        for u_domain in choice_domain_list[i]:  # 해당 유저가 선택한 domain 에 해당하는 데이터 가져오기
            user_idxs = set(np.random.choice(all_idxs[u_domain], num_items[u_domain], replace=False))
            dict_users.setdefault(i, set()).add(tuple(user_idxs))
            all_idxs[u_domain] = list(set(all_idxs[u_domain]).difference(user_idxs))

    return dict_users

def mnist_noniid_unequal(dataset, num_users):
    '''
def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
'''
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    # num_shards, num_imgs = 1200, 50
    # idx_shard = [i for i in range(num_shards)]
    # dict_users = {i: np.array([]) for i in range(num_users)}
    # idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    #
    # # sort labels
    # idxs_labels = np.vstack((idxs, labels))
    # idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    # idxs = idxs_labels[0, :]
    #
    # # Minimum and maximum shards assigned per client:
    # min_shard = 1
    # max_shard = 30
    #
    # # Divide the shards into random chunks for every client
    # # s.t the sum of these chunks = num_shards
    # random_shard_size = np.random.randint(min_shard, max_shard+1,
    #                                       size=num_users)
    # random_shard_size = np.around(random_shard_size /
    #                               sum(random_shard_size) * num_shards)
    # random_shard_size = random_shard_size.astype(int)
    #
    # # Assign the shards randomly to each client
    # if sum(random_shard_size) > num_shards:
    #
    #     for i in range(num_users):
    #         # First assign each client 1 shard to ensure every client has
    #         # atleast one shard of data
    #         rand_set = set(np.random.choice(idx_shard, 1, replace=False))
    #         idx_shard = list(set(idx_shard) - rand_set)
    #         for rand in rand_set:
    #             dict_users[i] = np.concatenate(
    #                 (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
    #                 axis=0)
    #
    #     random_shard_size = random_shard_size-1
    #
    #     # Next, randomly assign the remaining shards
    #     for i in range(num_users):
    #         if len(idx_shard) == 0:
    #             continue
    #         shard_size = random_shard_size[i]
    #         if shard_size > len(idx_shard):
    #             shard_size = len(idx_shard)
    #         rand_set = set(np.random.choice(idx_shard, shard_size,
    #                                         replace=False))
    #         idx_shard = list(set(idx_shard) - rand_set)
    #         for rand in rand_set:
    #             dict_users[i] = np.concatenate(
    #                 (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
    #                 axis=0)
    # else:
    #
    #     for i in range(num_users):
    #         shard_size = random_shard_size[i]
    #         rand_set = set(np.random.choice(idx_shard, shard_size,
    #                                         replace=False))
    #         idx_shard = list(set(idx_shard) - rand_set)
    #         for rand in rand_set:
    #             dict_users[i] = np.concatenate(
    #                 (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
    #                 axis=0)
    #
    #     if len(idx_shard) > 0:
    #         # Add the leftover shards to the client with minimum images:
    #         shard_size = len(idx_shard)
    #         # Add the remaining shard to the client with lowest data
    #         k = min(dict_users, key=lambda x: len(dict_users.get(x)))
    #         rand_set = set(np.random.choice(idx_shard, shard_size,
    #                                         replace=False))
    #         idx_shard = list(set(idx_shard) - rand_set)
    #         for rand in rand_set:
    #             dict_users[k] = np.concatenate(
    #                 (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
    #                 axis=0)
    #
    # return dict_users
def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users
def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.train_labels)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
