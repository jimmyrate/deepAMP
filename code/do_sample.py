import os, sys
import torch
import numpy as np
import random
import fixed_parameters as FP
from dataPre import train_conf, model_conf, sample_conf
from torch.utils.data.dataloader import DataLoader
from src.DataProcessTools import MaskCollate
from src.utils import checkDirs, stripTAG
from rwHelper import *
from alive_progress import alive_bar

def heteroSample(device,att_model,sample_loader,itos,sample_times,isRandom, ForceRandom,isPair,mask_ratio,save_path, src_data=None):
    att_model.eval()
    pairCabin = list()
    max_pred = 0
    with torch.no_grad():
        for st in range(sample_times): 
            print(f'Sample in the {st+1} time...')
            with alive_bar(len(sample_loader)) as bar:
                for batch_idx, (masked_seqs, masked_tokens, masked_pos) in enumerate(sample_loader):
                    masked_seqs = masked_seqs.to(device)
                    masked_pos = masked_pos.to(device)
                    masked_tokens = masked_tokens.to(device)
                    logits, _ = att_model(masked_seqs, masked_pos)
                    h = logits[:,-1,:]
                    btz, max_pred, vocab_size = logits.size()
                    if isRandom:
                        if isPair and not ForceRandom:
                            logits = logits + torch.randn(btz, max_pred, vocab_size).to(device)
                        else:
                            logits = torch.randn(btz, max_pred, vocab_size).to(device)
                        logits[:,-1,:] = h
                    # apply softmax to convert to probabilities
                    probs = torch.softmax(logits, dim=-1)             #[batch, max_pred, vocabsize]
                    # sample from the distribution or take the most likely
                    p_max, pred_token = torch.topk(probs, k=1, dim=-1)
                    pred_token = pred_token.squeeze(-1)
                    x = masked_seqs.scatter(dim=1, index=masked_pos, src=masked_tokens)
                    y = masked_seqs.scatter(dim=1, index=masked_pos, src=pred_token)
                    x = x.cpu().numpy() if 'cpu' != d else x.numpy()
                    y = y.cpu().numpy() if 'cpu' != d else y.numpy()
                    x = [stripTAG(''.join([itos[k] for k in i])) for i in x]
                    y = [stripTAG(''.join([itos[k] for k in i])) for i in y]
                    # assert len(x) == len(y)
                    pairs = [y[i]+' '+x[i] for i in range(len(x)) if len(x[i]) == len(y[i]) and x[i]!=y[i]] 
                    pairCabin.extend(pairs)
                    bar()
            pairCabin = list(set(pairCabin))

    cleanCabin = list()
    for data in pairCabin:
        x, y = data.strip().split(' ')
        c =  sum([int(i!=j) for i,j in zip(list(x),list(y))])
        if c <= min(max_pred, max(1, int(len(y) * mask_ratio))):
            cleanCabin.append(data)
    pairCabin = cleanCabin
    
    if sample_conf.isPair:
        change = dict()
        for data in pairCabin:
            x, y = data.strip().split(' ')
            c =  sum([int(i!=j) for i,j in zip(list(x),list(y))])
            if c not in change:
                change[c] = 0
            change[c] += 1

        for c, n in sorted(change.items(), key = lambda x:x[0]):
            print(f'{n} pairs changed in {c} points')
            
        lineTxtHelper(save_path).writeLines(pairCabin)
        print(f'Resulted in {len(pairCabin)} pairs, have been saved in {save_path}')
        return pairCabin
    else:
        singleCabin = [i.split(' ')[0] for i in pairCabin]
        singleCabin = list(set(singleCabin))
        if src_data is not None:
            singleCabin = list(set(singleCabin)- set(src_data))
        lineTxtHelper(save_path).writeLines(singleCabin)
        print(f'Resulted in {len(singleCabin)} samples, have been saved in {save_path}')
        return singleCabin

        


if __name__ == "__main__":
    att_model = train_conf.model
    sample_dataset = sample_conf.dataset
    d = 'cpu'
    device = None
    if torch.cuda.is_available():
        d = train_conf.device
        if d == 'cuda:multi':
            device = torch.cuda.current_device()
            att_model = torch.nn.DataParallel(att_model).to(device)
        else:
            device = torch.device(d)
            att_model = att_model.to(device)

    print('The code uses ' + d)
    print(train_conf.train_attr)
    pin_memory = False if d=='cpu' else True
    sample_loader = DataLoader(sample_dataset, 
            shuffle=True, pin_memory=pin_memory,batch_size=sample_conf.batch_size,num_workers=sample_conf.num_workers, drop_last=False,
            collate_fn=MaskCollate(sample_dataset,mask=sample_conf.mask,mask_ratio=sample_conf.mask_ratio,max_pred=sample_conf.max_pred,sample=False))
    itos = sample_conf.dataset.itos

    modeldir = train_conf.basicPath
    dir_list = sorted([i for i in os.listdir(modeldir) if i.startswith('epoch_')], key=lambda s:int(s.split('_')[1]))
    dir_list_star = ['epoch_'+str(i-1) for i in FP.SPECULATED_EPOCH]
    dir_list = list(set(dir_list) & set(dir_list_star))
    dir_list = sorted(dir_list,key = lambda s:int(s.split('_')[1]))
    print(f'isRandom:{sample_conf.isRandom}, isPair:{sample_conf.isPair} ,sample times:{sample_conf.sample_times}, batchsize:{sample_conf.batch_size}, mask_ratio:{sample_conf.mask_ratio}, max_pred:{sample_conf.max_pred}')
    print(f'Epochs:{dir_list}')
    for ep in dir_list:
        print(ep+':')
        # if ep.startswith('epoch_'):
        in_path = f'{modeldir}{ep}/'
        ckpt_path = in_path + 'model.pkl'
        data_path = checkDirs(in_path + 'data/')
        sample_polypeptide_path = checkDirs(data_path + 'Sample_polypeptide/')

        file_name = f'daset={sample_conf.daset},sample_times={sample_conf.sample_times},batchsize={sample_conf.batch_size},mask_ratio={sample_conf.mask_ratio},max_pred={sample_conf.max_pred},isPair={sample_conf.isPair},isRandom={sample_conf.isRandom}.txt'
        save_path = f'{sample_polypeptide_path}{file_name}'

        att_model.load_state_dict(torch.load(ckpt_path, map_location = lambda storage, loc:storage))
        att_model = att_model.to(device)
        
        output= heteroSample(device=device,
                        att_model=att_model,
                        sample_loader=sample_loader,
                        itos=itos,
                        sample_times=sample_conf.sample_times,
                        isRandom=sample_conf.isRandom,
                        ForceRandom=sample_conf.ForceRandom,
                        isPair=sample_conf.isPair,
                        mask_ratio=sample_conf.mask_ratio,
                        save_path=save_path)

        if sample_conf.outside_record:
            sample_conf.alterAttr(sample_model=ckpt_path)
            outsidePath = checkDirs(sample_conf.outsidePath)
            file_name = f'daset={sample_conf.daset},sample_times={sample_conf.sample_times},batchsize={sample_conf.batch_size},mask_ratio={sample_conf.mask_ratio},max_pred={sample_conf.max_pred},isPair={sample_conf.isPair},isRandom={sample_conf.isRandom}.txt'
            save_file = f'{outsidePath}{file_name}'
            lineTxtHelper(save_file).writeLines(output)
            print(f'Outside path:{save_file}')
        print()



