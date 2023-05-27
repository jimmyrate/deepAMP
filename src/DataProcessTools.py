import torch
import random
import numpy as np
import fixed_parameters as FP

class MaskCollate:
    def __init__(self, dataset, ST=FP.SPACIAIL_TAG, mask_ratio=0.3, max_pred=20, mask=True, sample=False):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dataset = dataset
        self.ST = ST
        self.mask_ratio = mask_ratio
        self.max_pred = max_pred
        self.mask = mask
        self.sample = sample

    def mask_collate(self, rdata):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        maxlen = self.dataset.block_size
        max_pred = self.max_pred
        mask_ratio = self.mask_ratio
        stoi = self.dataset.stoi
        itos = self.dataset.itos
        vocab_size = self.dataset.vocab_size
        
        batch = list()
        for _, data in enumerate(rdata):
            input_ids, target_ids = None, None
            # if len(data[0]) <=1:
            if not isinstance(data[0],(list, np.ndarray)):
                input_ids = data
            else:
                input_ids, target_ids = data[0], data[1]
            n_pred =  min(max_pred, max(1, int(len(input_ids) * mask_ratio))) # mask_ratio % of tokens in one sentence
            if self.sample:
                n_pred = random.randint(1,n_pred)
            cand_maked_pos = [i for i, token in enumerate(input_ids)
                            if token != stoi[self.ST['[CLS]']] and token != stoi[self.ST['[SEP]']] ] # candidate masked position
            random.shuffle(cand_maked_pos)
            masked_tokens, masked_pos = [], []
            for pos in cand_maked_pos[:n_pred]:
                masked_pos.append(pos)

                if target_ids is None:
                    masked_tokens.append(input_ids[pos])
                else:
                    masked_tokens.append(target_ids[pos])

                if self.mask:
                    if random.random() < 0.8:  # 80%
                        input_ids[pos] = stoi[self.ST['[MASK]']] # make mask
                    elif random.random() >= 0.9:  # 10%
                        index = random.randint(0, vocab_size - 1) # random index in vocabulary
                        while index == stoi[self.ST['[CLS]']] or index == stoi[self.ST['[SEP]']] or index == stoi[self.ST['[PAD]']]: # can't involve 'CLS', 'SEP', 'PAD'
                            index = random.randint(0, vocab_size - 1)
                            input_ids[pos] = index # replace
            # Paddings
            n_pad = maxlen - len(input_ids)
            input_ids.extend([ stoi[self.ST['[PAD]']] ] * n_pad)
            if max_pred > n_pred:
                n_pad = max_pred - n_pred
                masked_tokens.extend([ 0 ] * n_pad)
                # masked_tokens.extend([stoi[self.ST['[CLS]']]] * n_pad)
                masked_pos.extend([ 0 ] * n_pad)
            batch.append([input_ids, masked_tokens, masked_pos])
        # masked_seqs, masked_tokens, masked_pos = zip(*batch)
        masked_seqs, masked_tokens, masked_pos = map(torch.LongTensor,zip(*batch))
        return masked_seqs, masked_tokens, masked_pos

    def __call__(self, batch):
        return self.mask_collate(batch)

