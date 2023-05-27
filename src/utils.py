import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
import fixed_parameters as FP
from torch.nn import functional as F
from torch.optim.lr_scheduler import CyclicLR
from src.warmupScheduler import warmupLR
from src.Radam import *
from src.lookahead import Lookahead
from src.DataProcessTools import MaskCollate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def top_k_logits_batch(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:,:, [-1]]] = -float('Inf')
    return out

def getVocab(path):
    vocab = open(path, 'r').read().strip()
    vocab = list(vocab)
    return vocab

def checkDirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def stringToindex(line, stoi):
    one = list()
    one.append(FP.SPACIAIL_TAG['[CLS]'])
    one.extend(line)
    one.append(FP.SPACIAIL_TAG['[SEP]'])
    one = [stoi[i] for i in one]
    return one

def stripTAG(line:str):
    line = line.replace(FP.SPACIAIL_TAG['[CLS]'],'')
    line = line.replace(FP.SPACIAIL_TAG['[SEP]'],'')
    line = line.replace(FP.SPACIAIL_TAG['[PAD]'],'')
    line = line.replace(FP.SPACIAIL_TAG['[MASK]'],'')
    return line



def getTextList(path):
    textlist = list()
    with open(path, 'r') as f:
        text = f.read().strip().split('\n')
        for line in text:
            one = list()
            one.append(FP.SPACIAIL_TAG['[CLS]'])
            one.extend(line)
            one.append(FP.SPACIAIL_TAG['[SEP]'])
            textlist.append(one)
    return textlist

def StrListToIdxList(contexts, stoi):
    textlist = list()
    text = contexts
    for line in text:
        one = list()
        # one.append(FP.SPACIAIL_TAG['[CLS]'])
        # one.extend(line)
        # one.append(FP.SPACIAIL_TAG['[SEP]'])
        one.append(stoi[FP.SPACIAIL_TAG['[CLS]']])
        one.extend([stoi[s] for s in line])
        one.append(stoi[FP.SPACIAIL_TAG['[SEP]']])
        textlist.append(one)
    return textlist

def getPairTextList(path):
    textlist = list()
    with open(path, 'r') as f:
        text = f.read().strip().split('\n')
        for line in text:
            one = list()
            a, b = line.split(' ')
            la = list()
            la.append(FP.SPACIAIL_TAG['[CLS]'])
            la.extend(a)
            la.append(FP.SPACIAIL_TAG['[SEP]'])
            lb = list()
            lb.append(FP.SPACIAIL_TAG['[CLS]'])
            lb.extend(b)
            lb.append(FP.SPACIAIL_TAG['[SEP]'])
            one.append(la)
            one.append(lb)
            textlist.append(one)
    return textlist

def divideData(full_data: list, ratio: float, shuffle=True):
    datalist = full_data
    n_train = int(ratio*len(datalist))
    if shuffle: 
        random.shuffle(datalist)
    train_list, test_list = datalist[:n_train], datalist[n_train:]
    return train_list, test_list

def save_checkpoint(model, ckpt_path):
    # DataParallel wrappers keep raw model object in .module attribute
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(raw_model.state_dict(), ckpt_path)
    
def save_record(p, m, path):
    '''
     p ['build', 'add']
    '''
    if p == 'build':
        with open(path, 'w') as f:
            f.write(m)
    else:
        with open(path, 'a') as f:
            f.write(m)

def configure_optimizers(model, method, learning_rate, weight_decay):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear,)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding,)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if 'sigma' in fpn:
                no_decay.add(fpn)

            if (pn.endswith('bias') or pn.startswith('bias')):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif (pn.endswith('weight') or pn.startswith('weight')) and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif (pn.endswith('weight') or pn.startswith('weight')) and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # special case the position embedding parameter in the root GPT module as not decayed
    no_decay.add('pos_embed')

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    optimizer = None
    if method == 'Adam':
        optimizer = torch.optim.Adam(optim_groups,lr=learning_rate)
    elif method == 'AdamW':
        optimizer = torch.optim.AdamW(optim_groups,lr=learning_rate)
    elif method == 'SDG':
        optimizer = torch.optim.SDG(optim_groups,lr=learning_rate)
    elif method == 'Ranger':
        optimizer_inner = RAdam(optim_groups,lr=learning_rate)
        optimizer = Lookahead(optimizer_inner, k=5, alpha=0.5)
    return optimizer

def configure_scheduler(optimizer, train_conf, steps_per_epoch):
    scheduler_ = train_conf.lr_scheduler
    scheduler = None
    if scheduler_ == 'CyclicLR':
        if train_conf.warmup_epochs[0] < 1.0:
            step_size_up = int(steps_per_epoch*train_conf.warmup_epochs[0])
            step_size_down = steps_per_epoch - step_size_up
        else:
            step_size_up = int(steps_per_epoch*train_conf.warmup_epochs[0])
            step_size_down = step_size_up
        scheduler = CyclicLR(optimizer=optimizer,
                    base_lr=train_conf.init_lr*2,
                    max_lr=train_conf.max_lr*2,
                    step_size_up=step_size_up,
                    step_size_down=step_size_down,
                    mode='exp_range',
                    gamma=0.99991,
                    scale_fn=None,
                    scale_mode='cycle',
                    cycle_momentum=False,
                    base_momentum=0.8,
                    max_momentum=0.9,
                    last_epoch=-1)

    if scheduler_ == 'WarmupLR':
        scheduler = warmupLR(
                    optimizer=optimizer,
                    warmup_epochs=train_conf.warmup_epochs*2,
                    total_epochs=train_conf.total_epochs*2,
                    steps_per_epoch=steps_per_epoch,
                    init_lr=train_conf.init_lr*2,
                    max_lr=train_conf.max_lr*2,
                    final_lr=train_conf.final_lr*2)
    if scheduler is None:
        raise RuntimeError(f'{scheduler_} is not a available scheduler')
    return scheduler

def top_k_top_p_filtering(logits, top_k=0, top_p=0.99, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

class top_logits_func:
    def __init__(self,top_k=None,top_p=None, min_tokens_to_keep=None):
        self.top_k = 0 if top_k is None else top_k
        self.top_p = 1.0 if top_p is None else top_p
        self.min_tokens_to_keep = 1 if min_tokens_to_keep is None else min_tokens_to_keep
    
    def __call__(self,logits):
        return top_k_top_p_filtering(logits, top_k=self.top_k, top_p=self.top_p, min_tokens_to_keep=self.min_tokens_to_keep)

class Experience:
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores."""
    def __init__(self, max_size=100):
        self.memory = []
        self.max_size = max_size

    def add_experience(self, experience):
        """Experience should be a list of (masked_seq, masked_tokens, masked_pos, pred_tokens, y, score) tuples"""
        self.memory.extend(experience)
        self.memory.sort(key = lambda x: x[-1], reverse=True)
        
        if len(self.memory)>self.max_size:
            # Remove duplicates
            idxs, IDs = [], {}
            for i, exp in enumerate(self.memory):
                # ID = (exp[2].tolist(),exp[3].tolist())
                ID = exp[4].tolist()
                ID2 = (exp[2].tolist(),exp[3].tolist())
                if ID not in IDs:
                    idxs.append(i)
                    # IDs.append(ID)
                    IDs[ID] = []
                    IDs[ID].append(ID2)
                elif (len(IDs[ID])<10) and (ID2 not in IDs[ID]):
                    idxs.append(i)
                    IDs[ID].append(ID2)

            self.memory = [self.memory[idx] for idx in idxs]
            # Retain highest scores
            self.memory.sort(key = lambda x: x[-1], reverse=True)
            self.memory = self.memory[:self.max_size]
            best_score = round(self.memory[0][-1],2)
            avg_experience_score = round(np.array([i[-1] for i in self.memory]).mean(),2)
            # print("\nBest score in memory: {:.2f}".format(best_score))
            return best_score, avg_experience_score
        return None, None

    def sample(self, n):
        """Sample a batch size n of experience"""
        if len(self.memory)<n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[-1] for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=np.exp(scores)/np.exp(scores).sum() )
            # sample = np.random.choice(len(self), size=n, replace=False )
            sample = [self.memory[i] for i in sample]
            masked_seqs, masked_tokens, masked_pos, pred_tokens, y, scores = zip(*sample)
        return masked_seqs, masked_tokens, masked_pos, pred_tokens, y, scores

    def __len__(self):
        return len(self.memory)