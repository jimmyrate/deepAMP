import os, sys
import torch
import numpy as np
import random
import fixed_parameters as FP
from dataPre import train_conf, model_conf, sample_conf, reinforce_conf
from torch.utils.data.dataloader import DataLoader
from src.utils import checkDirs
from rwHelper import *


# def getVotedUnion(data, vote=2):
#     count = dict()
#     for s in data:
#         if s not in count:
#             count[s] = 0
#         count[s] += 1
#     union_data = [k for k,v in count.items() if v>=vote]
#     union_data = list(set(union_data))
#     return union_data

class getVotedUnion:
    def __init__(self, vote=2):
        self.vote = vote

    def make(self, data):
        vote = self.vote
        count = dict()
        for s in data:
            if s not in count:
                count[s] = 0
            count[s] += 1
        union_data = [k for k, v in count.items() if v >= vote]
        union_data = list(set(union_data))
        return union_data

    def __call__(self, data):
        return self.make(data)


if __name__ == "__main__":
    att_model = train_conf.model
    d = 'cpu'
    device = torch.device(d)
    if torch.cuda.is_available():
        d = train_conf.device
        device = torch.device(d)
        att_model = att_model.to(device)
    print('Generate code uses ' + d)
    print(train_conf.train_attr)

    src_data = None
    if reinforce_conf.src_daset is not None:
        src_data = lineTxtHelper(train_conf.protein_path[reinforce_conf.src_daset]).readLines()
        src_data = [i.strip() for i in src_data]
        src_data = list(set(src_data))

    modeldir = train_conf.basicPath
    package_path = checkDirs(f'{modeldir}Sample_polypeptide_package/')
    dir_list = sorted([i for i in os.listdir(modeldir) if i.startswith('epoch_')], key=lambda s: int(s.split('_')[1]))
    dir_list_star = ['epoch_' + str(i - 1) for i in FP.SPECULATED_EPOCH]
    dir_list = list(set(dir_list) & set(dir_list_star))
    dir_list = sorted(dir_list, key=lambda s: int(s.split('_')[1]))
    print(
        f'isRandom:{sample_conf.isRandom}, isPair:{sample_conf.isPair} ,sample times:{sample_conf.sample_times}, batchsize:{sample_conf.batch_size}, mask_ratio:{sample_conf.mask_ratio}, max_pred:{sample_conf.max_pred}')
    print(f'Epochs:{dir_list}')

    package = dict()
    for ep in dir_list:
        in_path = f'{modeldir}{ep}/'
        ckpt_path = in_path + 'model.pkl'
        data_path = checkDirs(in_path + 'data/')
        sample_polypeptide_path = checkDirs(data_path + 'Sample_polypeptide/')
        for p_txt in os.listdir(sample_polypeptide_path):
            if p_txt.endswith('config.txt'):
                continue
            method = p_txt.split('/')[-1]
            method = method.rsplit('.', 1)[0]
            if method not in package:
                package[method] = list()
            package[method].extend(lineTxtHelper(f'{sample_polypeptide_path}{p_txt}').readLines())

    # package = dict(zip(package.keys(),map(lambda x:list(set(x)), package.values())) )
    vote = 2
    if sample_conf.isRandom or len(dir_list) < 2:
        vote = 1
    package = dict(zip(package.keys(), map(getVotedUnion(vote), package.values())))
    for name, data in package.items():
        # data = data[1:] if len(data[0])==0 else data
        data = [i for i in data if len(i) > 0]
        if src_data is not None:
            data = list(set(data) - set(src_data))
        save_file = f'{package_path}{name}.txt'
        lineTxtHelper(save_file).writeLines(data)
        print(f'Saved "{save_file}", len {len(data)}')
    package_message_file = f'{package_path}pakage_message.txt'
    lineTxtHelper(package_message_file).writeLines(dir_list)
