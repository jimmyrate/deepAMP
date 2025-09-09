import pickle
import numpy as np
from rwHelper import *
from tqdm import tqdm

class s2t(object):    
    def __init__(self, filename):
        self.t2v = {}
        self.dim = None
        for line in open(filename):
            line = line.strip('\n')
            line = line.split('\t')
            t = line[0]         
            v = np.array([float(x) for x in line[1].split()])
            if self.dim is None:
                self.dim = len(v)
            else:
                v = v[:self.dim]
            self.t2v[t] = v
    
    def embed(self, seq):
        if seq.find(' ') > 0:
            s = seq.strip().split()
        else:
            s = list(seq.strip())
        
        rst = []
        for x in s:
            v = self.t2v.get(x)
            if v is None:
                continue
            rst.append(v)
        return np.array(rst)
    
    def embed_normalized(self, seq, length=60):
        rst = self.embed(seq)
        if len(rst) > length:
            return rst[:length]
        elif len(rst) < length:
            return np.concatenate((rst, np.zeros((length - len(rst), self.dim))))
        return rst

    def embed_RECM_position(self,seq):
        if seq.find(' ') > 0:
            s = seq.strip().split()
        else:
            s = list(seq.strip())
        order_char = ['A','C','D','E',
                      'F','G','H','I',
                      'K','L','M','N',
                      'P','Q','R','S',
                      'T','V','W','Y']
        rst = []
        for x in order_char:
            x_repeat = s.count(x)
            if x_repeat == 0:
                v = np.zeros(20)
            else:
                v = self.t2v.get(x)
                v = v * x_repeat
            v = v.tolist()
            rst = rst+v
        #rst = np.array(rst)

        #rst = rst.flatten()
        return np.array(rst)

def seq_to_list(seq_list):
    seq_arrays = []
    seq2t = s2t('src/RECM-position.txt')
    for seq in seq_list:
        seq_array = seq2t.embed_RECM_position(seq)
        #seq_array.flatten()
        seq_arrays.append(seq_array)
    return seq_arrays

def seq_to_feature(seq):

    seq2t = s2t('src/RECM-position.txt')
    seq_array = seq2t.embed_RECM_position(seq)
    return seq_array

def seq_score(svm_path,seq_list):
    svm_model = objHelper(svm_path).load_obj()
    seq_array = seq_to_list(seq_list)
    prediction_proba = svm_model.predict_proba(seq_array)
    prediction_score = [x[1] for x in prediction_proba]

    return prediction_score

def seq_score_dict(svm_path,seq_list):
    predict_score_dict = {}
    svm_model = objHelper(svm_path).load_obj()
    count = 0
    for seq in tqdm(seq_list):
        seq_array = seq_to_feature(seq)
        prediction_proba = svm_model.predict_proba(seq_array.reshape(1,-1))
        prediction_score = [x[1] for x in prediction_proba]
        predict_score_dict[seq] = prediction_score[0]
        count = count + 1
    return predict_score_dict


if __name__ == '__main__':
    svm_pene = '/home/lolo/lolo-bak/Code/AMP-GPT/AMP_SVM/model_pkl/svr_grid_outcomestrain_20210617_py38.pkl'
    svm_amp = '/home/lolo/lolo-bak/Code/AMP-GPT/AMP_SVM/model_pkl/svr_model_20210115_01.pkl'

    # seq_path = 'output/RL/daset=random_chem_penetritin_high_activity_all_pair,seed=6,optimizer=Ranger,batch_size=32,epochs=200,lr=1e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15_epoch_199/daset=chem_round1_best,seed=6,optimizer=Ranger,batch_size=32,epochs=500,lr=1e-06,weight_decay=1e-06,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=50/reinforcement_proteins_package/RQIKIWFQWRRWKWKK_num_10.txt'
    # seq_List = lineTxtHelper(seq_path).readLines()

    seq_path = 'output/finetune/daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=random_pair,seed=3,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/Sample_polypeptide_package/daset=inter_test_SKVWRHWRRFWHRAHRKK,sample_times=20,batchsize=5,mask_ratio=0.3,max_pred=4,isPair=False,isRandom=False.txt'
    seq_List = lineTxtHelper(seq_path).readLines()
    # seq_List = ['VKRWKKWKRKWKKWV','SKVWRHWRRFWHRAHRKK',]
    score_list = seq_score(svm_amp,seq_List)
    seqs_score = list(zip(seq_List, score_list))
    seqs_score = sorted(seqs_score, key=lambda one:one[1], reverse=True)
    seqs_score = [{'seqs':line[0], 'sorce':line[1]} for line in seqs_score]
    out_path = seq_path.rsplit('.', maxsplit=1)[0] + '.csv'
    out_path = seq_path.rsplit('/', 1)[-1]
    out_path = 'workplace/feng/'+ out_path
    csvHelper(out_path).writeRows(seqs_score)

    # out_path = seq_path.rsplit('.', maxsplit=1)[0] + '.csv'
    # csvHelper(out_path).writeRows(seqs_score)
    print(seqs_score)

    