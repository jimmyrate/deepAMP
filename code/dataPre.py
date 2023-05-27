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
seed = 3
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

text_path = {'uniport_all_60':'data/uni_all_60.txt','uniport_60_polypeptide':'data/uniprot_10_to_60.txt',
            'E.coli-AMP-best':'data/E.coli-AMP-best.txt','E.coli-AMP-better':'data/E.coli-AMP-better.txt',
            'E.coli-AMP-seq':'data/E.coli-AMP-seq.txt',
            'E.coli-AMP-normal':'data/E.coli-AMP-normal.txt',
            'E.coli-AMP-normal-last-100':'data/E.coli-AMP-normal-last-100.txt',
            'E.coli-AMP-normal-last-100-iter1-best5':'plot/general_model_used/e.coli-last-100-optimized-iter1-best5.txt',
            'SVM_TEST':'testdata/SVM_TEST_amp_seq.txt',
            'SVM_TEST_polypeptide':'testdata/SVM_TEST_polypeptide_seq.txt',
            'SVM_TEST_all':'testdata/SVM_TEST_all_seq.txt',
            'SVM_TEST_amp_low':'testdata/SVM_TEST_amp_seq_low.txt',
            'chem10':'data/chem/chem_10.txt',
            'chem20':'data/chem/chem_20.txt',
            'chem41':'data/chem/chem_all.txt',
            'chem_penetratin':'data/penetratin/penetratin_all.txt',
            'chem_penetritin_high_activity':'data/penetratin/penetritin_high_activity.txt',
            'chem_penetritin_high_activity_all':'data/penetratin/penetritin_high_activity_all.txt',
            'chem_penetritin_with_1_experiment':'data/penetratin/outcomes/penetratin_with_1_experiment.txt',
            'chem_penetritin_only_1_experiment':'data/penetratin/outcomes/penetratin_only_1_experiment.txt',
            'chem_round1_generated':'data/penetratin/outcomes/round1generated.txt',
            'chem_round1_best':'data/penetratin/outcomes/round1_best.txt',
            # 'AMP_pair':'output/Sample_polypeptide/output_pretrain_daset=uniport_60,optimizer=Ranger,batch_size=128,epochs=850,lr=0.0006,weight_decay=0,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=20_epoch_270.txt'
            # 'AMP_pair':'output/Sample_polypeptide/output_pretrain_daset=uniport_all_60,optimizer=Ranger,batch_size=128,epochs=350,lr=0.0006,weight_decay=0.01,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=20_epoch_215.txt',
            # 'AMP_pair':'output/Sample_polypeptide/pretrain_daset=uniport_all_60,optimizer=Ranger,batch_size=128,epochs=350,lr=0.0006,weight_decay=0.01,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=20_epoch_349/Sample_times=15,batchsize=10,mask_ratio=0.3,max_pred=20.txt'
            # 'AMP_pair':'output/Sample_polypeptide/pretrain_daset=uniport_all_60,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0001,weight_decay=1e-05,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=20_epoch_118/Sample_times=200,batchsize=10,mask_ratio=0.3,max_pred=4.txt'
            # 'AMP_pair':'output/Sample_polypeptide/pretrain_daset=uniport_all_60,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0001,weight_decay=1e-05,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=20_epoch_206/Sample_times=200,batchsize=10,mask_ratio=0.3,max_pred=4,isPair=True.txt'
            # 'AMP_pair':'output/Sample_polypeptide/pretrain_daset=uniport_all_60,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_2/Sample_times=200,batchsize=10,mask_ratio=0.3,max_pred=4,isPair=True.txt'
            'stimulate_pair':'output/Sample_polypeptide/pretrain_daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/Sample_times=100,batchsize=10,mask_ratio=0.3,max_pred=20,isPair=True,isRandom=False.txt',
            'homologous_pair':'pairsData/sw_500_best_pairs.txt',
            'random_pair':'output/Sample_polypeptide/pretrain_daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/Sample_times=100,batchsize=10,mask_ratio=0.3,max_pred=20,isPair=True,isRandom=True.txt',
            'random_pair_5w':'output/Sample_polypeptide/pretrain_daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=E.coli-AMP-best,sample_times=200,batchsize=10,mask_ratio=0.3,max_pred=20,isPair=True,isRandom=True.txt',
            'random_better_pair':'output/Sample_polypeptide/pretrain_daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=E.coli-AMP-better,sample_times=50,batchsize=10,mask_ratio=0.3,max_pred=20,isPair=True,isRandom=True.txt',
            'random_chem10_pair':'output/Sample_polypeptide/pretrain_daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=chem10,sample_times=50,batchsize=10,mask_ratio=0.3,max_pred=20,isPair=True,isRandom=True.txt',
            'random_chem20_pair':'output/Sample_polypeptide/pretrain_daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=chem20,sample_times=50,batchsize=10,mask_ratio=0.3,max_pred=20,isPair=True,isRandom=True.txt',
            'random_chem41_pair':'output/Sample_polypeptide/pretrain_daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=chem41,sample_times=50,batchsize=10,mask_ratio=0.3,max_pred=20,isPair=True,isRandom=True.txt',
            'random_chem_penetratin_pair':'output/Sample_polypeptide/pretrain_daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=chem_penetratin,sample_times=50,batchsize=10,mask_ratio=0.3,max_pred=20,isPair=True,isRandom=True.txt',
            'random_chem_penetritin_high_activity_pair':'output/Sample_polypeptide/pretrain_daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=chem_penetritin_high_activity,sample_times=50,batchsize=10,mask_ratio=0.3,max_pred=20,isPair=True,isRandom=True.txt',
            'random_chem_penetritin_high_activity_all_pair':'output/Sample_polypeptide/pretrain_daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=chem_penetritin_high_activity_all,sample_times=50,batchsize=10,mask_ratio=0.3,max_pred=20,isPair=True,isRandom=True.txt',
            'random_chem_penetritin_with_1_experiment_pair':'data/penetratin/outcomes/daset=chem_penetritin_with_1_experiment,sample_times=50,batchsize=10,mask_ratio=0.3,max_pred=20,isPair=True,isRandom=True.txt',
            'random_chem_penetritin_with_1_experiment_without_Heirarchy_pair':'data/penetratin/outcomes/chem_penetritin_with_1_experiment&pep2pene.txt',
            'random_chem_penetritin_only_1_experiment_pair':'data/penetratin/outcomes/daset=chem_penetritin_only_1_experiment,sample_times=50,batchsize=10,mask_ratio=0.3,max_pred=20,isPair=True,isRandom=True&hierarchy&pep2pene.txt',
            }

vocab_path = {'General':'vocab/Protein.vocab'}

block_size = 64 # spatial extent of the model for its context
# daset = 'random_chem_penetritin_only_1_experiment_pair'
#daset = 'random_chem_penetritin_high_activity_all_pair'
daset = 'random_pair'

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
    daset='chem_penetritin_only_1_experiment',isPair=True, isRandom=True, sample_times=50, batch_size=10, protein_path=text_path,
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

    # text_path[f'inter_test_{reinforce_conf.optimize_amp}']=  f'{train_conf.basicPath}reinforcement_proteins_package/{reinforce_conf.optimize_amp}_best5.txt'
    # text_path[f'inter_test_{reinforce_conf.optimize_amp}_best5']=  f'{train_conf.basicPath}Sample_polypeptide_package/inter_test_{reinforce_conf.optimize_amp}_best5.txt'
    #text_path[f'inter_test_{reinforce_conf.optimize_amp}_best5_gen2']=  f'{train_conf.basicPath}Sample_polypeptide_package/inter_test_{reinforce_conf.optimize_amp}_best5_gen2.txt'
    text_path[f'inter_test_{reinforce_conf.optimize_amp}_all']=  f'{train_conf.basicPath}reinforcement_proteins_package/{reinforce_conf.optimize_amp}_num_10.txt'
    sample_conf = makeSampleConfig(
        daset=f'inter_test_{reinforce_conf.optimize_amp}_all',isPair=False, isRandom=False, sample_times=20, batch_size=5, protein_path=text_path,
        mask=mask, mask_ratio=0.3, max_pred=4, num_workers=5,outside_record=False)
    sample_dataset = polypeptideDataset(getTextList(text_path[sample_conf.daset]), vocab, block_size)
    sample_conf.alterAttr(dataset=sample_dataset)

    # sample_conf = makeSampleConfig(
    #     daset='E.coli-AMP-normal-last-100',isPair=True, isRandom=True, ForceRandom=True, sample_times=20, batch_size=5, protein_path=text_path,
    #     mask=mask, mask_ratio=0.3, max_pred=4, num_workers=5,outside_record=False)
    # sample_dataset = polypeptideDataset(getTextList(text_path[sample_conf.daset]), vocab, block_size)
    # sample_conf.alterAttr(dataset=sample_dataset)
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
# output/finetune/daset=random_pair,seed=6,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15_epoch_199/daset=random_chem_penetritin_high_activity_all_pair,seed=6,optimizer=Ranger,batch_size=32,epochs=200,lr=1e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/reinforcement_proteins_package/RQIKIWFQNRRMKWKK_num_10.txt
    # text_path[f'inter_test_{reinforce_conf.optimize_amp}']=  f'{train_conf.basicPath}reinforcement_proteins_package/{reinforce_conf.optimize_amp}_best5.txt'
    text_path[f'inter_test_{reinforce_conf.optimize_amp}_all']=  f'{train_conf.basicPath}reinforcement_proteins_package/{reinforce_conf.optimize_amp}_num_10.txt'
    sample_conf = makeSampleConfig(
        daset=f'inter_test_{reinforce_conf.optimize_amp}_all',isPair=False, isRandom=False, sample_times=20, batch_size=5, protein_path=text_path,
        mask=mask, mask_ratio=0.3, max_pred=4, num_workers=5,outside_record=False)
    sample_dataset = polypeptideDataset(getTextList(text_path[sample_conf.daset]), vocab, block_size)
    sample_conf.alterAttr(dataset=sample_dataset)
######################################################## re-finetune ####################################################################

######################################################## reinforce ####################################################################################################
if train_conf.mode == 'RL':
    '''
        Reinforcement Learning
    '''
    mask = False
    train_conf.alterAttr(
                        # pretrained_model = f'output/finetune/daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=random_pair,seed={seed},optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/epoch_199/model.pkl', # general model
                        pretrained_model = 'output/finetune/daset=random_pair,seed=6,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15_epoch_199/daset=random_chem_penetritin_high_activity_all_pair,seed=6,optimizer=Ranger,batch_size=32,epochs=200,lr=1e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/epoch_199/model.pkl', #round 1 model
                        device='cuda:0',max_epochs=500, batch_size=32, learning_rate=1e-6, weight_decay=1e-6,lr_decay=True, doSave=True, doTest=True,experience_replay=True,
                        model=model, train_dataset=train_dataset, test_dataset= test_dataset, optimizer='Ranger',lr_scheduler='WarmupLR', #{'WarmupLR','CyclicLR'}
                        mask=mask, mask_ratio=0.3, max_pred=20, num_workers=5, daset=daset,warmup_epochs=50, protein_path=text_path, seed=seed)

    reinforce_conf = ReinforceConfig(reinforce_times=10, 
                            mask=mask, mask_ratio=0.3, max_pred=20, batch_size=10, isRandom=False,
                            src_daset='chem_penetritin_with_1_experiment',
                            # optimize_amp = 'FFPIVGKLLSGLL',
                            # optimize_amp = 'FFPIVGKLLSGLF',     #G0
                            # optimize_amp = 'FFPIVKKLLSGLF',     #G1
                            # optimize_amp = 'FLPIVKKLLRGLF',     #G2
                            # optimize_amp = 'RQIKIWFQNRRMKWKK',    #penetratin
                            optimize_amp = 'RQIKIWFQWRRWKWKK', #penetratin round 1 best
                            )

    # text_path[f'inter_test_{reinforce_conf.optimize_amp}']=  f'{train_conf.basicPath}reinforcement_proteins_package/{reinforce_conf.optimize_amp}_best5.txt'
    # sample_conf = makeSampleConfig(
    #     daset=f'inter_test_{reinforce_conf.optimize_amp}',isPair=False, isRandom=False, sample_times=20, batch_size=5, protein_path=text_path,
    #     mask=mask, mask_ratio=0.3, max_pred=4, num_workers=5,outside_record=False)
    # sample_dataset = polypeptideDataset(getTextList(text_path[sample_conf.daset]), vocab, block_size)
    # sample_conf.alterAttr(dataset=sample_dataset)
######################################################## reinforce ####################################################################################################
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