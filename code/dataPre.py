import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from src.model import AMPBERT
from src.utils import set_seed, getVocab, divideData, getTextList, getPairTextList
from configer import BertConfig, TrainerConfig, ReinforceConfig, makeSampleConfig
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
seed = 6
set_seed(seed)

class polypeptideDataset(Dataset):

    def __init__(self, data:list, vocab:list, block_size):
        data_size, vocab_size = len(data), len(vocab)
        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = [ i for i in data if len(i)<=block_size and len(i)>=5]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        # encode every character to an integer
        chunk = [self.stoi[s] for s in chunk]
        return chunk

class pairDataset(Dataset):

    def __init__(self, data:list, vocab:list, block_size):
        data_size, vocab_size = len(data), len(vocab)
        self.stoi = { ch:i for i,ch in enumerate(vocab) }
        self.itos = { i:ch for i,ch in enumerate(vocab) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = [ i for i in data if len(i[0])==len(i[1]) and len(i[0])<=block_size and len(i[1])>=5]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        # encode every character to an integer
        src = [self.stoi[s] for s in chunk[0]]
        trg = [self.stoi[s] for s in chunk[1]]
        return src, trg

text_path = {'uniport_all_60':'.data/uni_all_60.txt',
            'random_pair':'.data/AOM-training-pairs.txt',
            'random_chem_penetratin_pair':'.data/POM-training-pairs.txt'
            }

vocab_path = {'General':'vocab/Protein.vocab'}

block_size = 64 # spatial extent of the model for its context

daset = 'random_pair'
# daset = 'uniport_all_60'

Reinforce_mode = False
mask = None
train_conf = TrainerConfig(mode='pretrain')
if 'uniport' not in daset:
    train_conf.alterAttr(mode='finetune')
# if 'chem' in daset:
#     train_conf.alterAttr(mode='re-finetune')

if Reinforce_mode:
    train_conf.alterAttr(mode='RL')
    daset = 'chem_round1_best'
    vocab = getVocab(vocab_path['General'])
    if 'pair' in daset:
        train_text = getPairTextList(text_path[daset])
        train_dataset = pairDataset(train_text, vocab, block_size)
    else:
        train_text = getTextList(text_path[daset])
        train_dataset = polypeptideDataset(train_text, vocab, block_size)
    test_dataset = None

elif train_conf.mode == 'pretrain' or 'pair' not in daset:
    mask = True
    text = getTextList(text_path[daset])
    train_text, test_text = divideData(text, 0.8)
    vocab = getVocab(vocab_path['General'])
    train_dataset, test_dataset = polypeptideDataset(train_text, vocab, block_size), polypeptideDataset(test_text, vocab, block_size)
else:
    mask = True
    text = getPairTextList(text_path[daset])
    train_text, test_text = divideData(text, 0.8)
    vocab = getVocab(vocab_path['General'])
    train_dataset, test_dataset = pairDataset(train_text, vocab, block_size), pairDataset(test_text, vocab, block_size)

model_conf = BertConfig(train_dataset.vocab_size, train_dataset.block_size,
                    n_embd=32, n_layers=12, n_heads=12, d_model=512, d_ff=512*2, d_k=64, d_v=64)
model = AMPBERT(model_conf)


######################################################## pretrain ####################################################################
mask = True
train_conf.alterAttr(device='cuda:0',max_epochs=350, batch_size=256, learning_rate=7e-4, weight_decay=1e-4,lr_decay=True, doSave=True, doTest=True,
                        model=model, train_dataset=train_dataset, test_dataset= test_dataset, optimizer='Ranger',lr_scheduler='WarmupLR', #{'WarmupLR','CyclicLR'}
                        mask=mask, mask_ratio=0.3, max_pred=20, num_workers=5, daset=daset,warmup_epochs=30, protein_path=text_path, seed=seed)


sample_conf = makeSampleConfig(
    daset='antifungal',isPair=True, isRandom=False, sample_times=50, batch_size=10, protein_path=text_path,
    mask=mask, mask_ratio=0.3, max_pred=20, num_workers=5,)
sample_dataset = polypeptideDataset(getTextList(text_path[sample_conf.daset]), vocab, block_size)
sample_conf.alterAttr(dataset=sample_dataset)
######################################################## pretrain ####################################################################




######################################################## finetune ####################################################################
if train_conf.mode == 'finetune' and 'chem' not in daset:
    mask = True
    train_conf.alterAttr(
                        pretrained_model = 'output/pretrain/daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30/epoch_349/model.pkl',
                        device='cuda:0',max_epochs=200, batch_size=128, learning_rate=6e-5, weight_decay=1e-3,lr_decay=True, doSave=True, doTest=True,
                        model=model, train_dataset=train_dataset, test_dataset= test_dataset, optimizer='Ranger',lr_scheduler='WarmupLR', #{'WarmupLR','CyclicLR'}
                        mask=mask, mask_ratio=0.3, max_pred=20, num_workers=5, daset=daset,warmup_epochs=15, protein_path=text_path, seed=seed)

    print(f'mask={mask}')
    reinforce_conf = ReinforceConfig(reinforce_times=10, 
                            mask=mask, mask_ratio=0.3, max_pred=20, batch_size=10, isRandom=False,
                            # optimize_amp = 'FFPIVGKLLSGLL',
                            # optimize_amp = 'FFPIVGKLLSGLF',     #G0
                            # optimize_amp = 'FFPIVKKLLSGLF',     #G1
                            # optimize_amp = 'FLPIVKKLLRGLF'      #G2
                            #optimize_amp = 'KFHLFKKILKGLF'      #G3
                            optimize_amp = 'RQIKIWFQNRRMKWKK',    #penetratin
                            # optimize_amp = 'VKRWKKWKRKWKKWV',     # A3
                            # optimize_amp = 'SKVWRHWRRFWHRAHRKK',  # Chensinin-1b
                            )

    text_path[f'inter_test_{reinforce_conf.optimize_amp}_all']=  f'{train_conf.basicPath}reinforcement_proteins_package/{reinforce_conf.optimize_amp}_num_10.txt'
    sample_conf = makeSampleConfig(
        daset=f'inter_test_{reinforce_conf.optimize_amp}_all',isPair=False, isRandom=False, sample_times=20, batch_size=5, protein_path=text_path,
        mask=mask, mask_ratio=0.3, max_pred=4, num_workers=5,outside_record=False)
    sample_dataset = polypeptideDataset(getTextList(text_path[sample_conf.daset]), vocab, block_size)
    sample_conf.alterAttr(dataset=sample_dataset)


######################################################## finetune ####################################################################




######################################################## re-finetune ####################################################################
if 'chem' in daset and not Reinforce_mode:
    mask = False
    train_conf.alterAttr(
                        pretrained_model = f'output/finetune/daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=random_pair,seed={seed},optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/epoch_199/model.pkl', # general model
                        # pretrained_model = 'output/finetune/daset=random_pair,seed=6,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15_epoch_199/daset=random_chem_penetritin_high_activity_all_pair,seed=6,optimizer=Ranger,batch_size=32,epochs=200,lr=1e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/epoch_199/model.pkl', #round 1 model
                        device='cpu',max_epochs=200, batch_size=32, learning_rate=1e-5, weight_decay=1e-3,lr_decay=True, doSave=True, doTest=True,
                        model=model, train_dataset=train_dataset, test_dataset= test_dataset, optimizer='Ranger',lr_scheduler='WarmupLR', #{'WarmupLR','CyclicLR'}
                        mask=mask, mask_ratio=0.3, max_pred=20, num_workers=5, daset=daset,warmup_epochs=15, protein_path=text_path, seed=seed)
    
    reinforce_conf = ReinforceConfig(reinforce_times=10,
                            mask=mask, mask_ratio=0.3, max_pred=20, batch_size=10, isRandom=False,
                            # optimize_amp = 'FFPIVGKLLSGLL',
                            # optimize_amp = 'FFPIVGKLLSGLF',     #G0
                            # optimize_amp = 'FFPIVKKLLSGLF',     #G1
                            # optimize_amp = 'FLPIVKKLLRGLF',     #G2
                            optimize_amp = 'RQIKIWFQNRRMKWKK',    #penetratin
                            # optimize_amp = 'RQIKIWFQWRRWKWKK', #penetratin round 1 best
                            )

    # text_path[f'inter_test_{reinforce_conf.optimize_amp}']=  f'{train_conf.basicPath}reinforcement_proteins_package/{reinforce_conf.optimize_amp}_best5.txt'
    text_path[f'inter_test_{reinforce_conf.optimize_amp}_all']=  f'{train_conf.basicPath}reinforcement_proteins_package/{reinforce_conf.optimize_amp}_num_10.txt'
    sample_conf = makeSampleConfig(
        daset='optimize_amp',isPair=False, isRandom=False, sample_times=50, batch_size=5, protein_path=text_path,
        mask=mask, mask_ratio=0.3, max_pred=4, num_workers=5,outside_record=False)
    sample_dataset = polypeptideDataset(getTextList(text_path[sample_conf.daset]), vocab, block_size)
    sample_conf.alterAttr(dataset=sample_dataset)
######################################################## re-finetune ####################################################################


if __name__ == "__main__":
    print(model)
    params = list(model.parameters())
    k = 0
    for i in params:
        l = 1
        for j in i.size():
            l *= j
        k = k + l
    print("Parameters：" + str(k))