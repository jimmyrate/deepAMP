import pandas as pd
import numpy as np
from rwHelper import *
from score_list import seq_score, seq_score_dict

data_path = {
    'chem': '/home/leo/fightinglee/AMP-Projects/AMP/AMP-GPT/test_value/data/only_chem1.txt',
    'chem_gen1': '/home/leo/fightinglee/AMP-Projects/evolvX-0.1.0/seq-data/Experiment/G1.txt',
    'chem_gen2': '/home/leo/fightinglee/AMP-Projects/evolvX-0.1.0/seq-data/Experiment/G2.txt',
    'chem_gen3': '/home/leo/fightinglee/AMP-Projects/evolvX-0.1.0/seq-data/Experiment/G3.txt',
    'model_gen1':'/home/leo/fightinglee/AMP-Projects/Protein-Bert/output/finetune/daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=random_pair,seed=3,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/Sample_polypeptide_package/daset=inter_test_FFPIVGKLLSGLF,sample_times=20,batchsize=5,mask_ratio=0.3,max_pred=4,isPair=False,isRandom=False.txt',
    'model_gen2':'/home/leo/fightinglee/AMP-Projects/Protein-Bert/output/finetune/daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=random_pair,seed=3,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/Sample_polypeptide_package/daset=inter_test_FFPIVGKLLSGLF_best5,sample_times=20,batchsize=5,mask_ratio=0.3,max_pred=4,isPair=False,isRandom=False.txt',
    'model_gen3':'/home/leo/fightinglee/AMP-Projects/Protein-Bert/output/finetune/daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=random_pair,seed=3,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/Sample_polypeptide_package/daset=inter_test_KFHLFKKILKGLF_best5_gen2,sample_times=20,batchsize=5,mask_ratio=0.3,max_pred=4,isPair=False,isRandom=False.txt',
    'random':'/home/leo/fightinglee/AMP-Projects/AMP/AMP-GPT/data/daset=random_pair,seed=6,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/reinforcement_proteins_package/FFPIVGKLLSGLF_num_10_random.txt_matrix_result.csv',
    'hdry_gen1':'/home/leo/fightinglee/AMP-Projects/Protein-Bert/data/hdryamp/hdyr_gen1.txt',
    'hdry_gen2': '/home/leo/fightinglee/AMP-Projects/Protein-Bert/data/hdryamp/hdyr_gen2.txt',
    'hdry_gen3': '/home/leo/fightinglee/AMP-Projects/Protein-Bert/data/hdryamp/hdyr_gen3.txt',
    'pepcvae_gen1':'/home/leo/fightinglee/AMP-Projects/Protein-Bert/data/hdryamp/pepcvae_gen1.txt',
    'pepcvae_gen2': '/home/leo/fightinglee/AMP-Projects/Protein-Bert/data/hdryamp/pepcvae_gen2.txt',
    'pepcvae_gen3': '/home/leo/fightinglee/AMP-Projects/Protein-Bert/data/hdryamp/pepcvae_gen3.txt'
}



def get_amp_list(path):
    seq_list = []
    with open(path,'r') as fr:
        for line in fr:
            line=line.strip('\n')
            seq_list.append(line)
    return seq_list

def get_order(path):
    dict_seq_order = {}
    i = 1
    with open(path,'r') as fr:
        for line in fr:
            line=line.strip('\n')
            dict_seq_order[line]=str(i)
            i+=1

    return dict_seq_order


def add_svm_scores_to_csv(input_csv_path, svm_model, output_csv_path):
    # 读取CSV文件
    # df = pd.read_csv(input_csv_path, usecols=['seq', 'MIC'])

    df = pd.read_csv(input_csv_path, usecols=['seq'])
    # df2 = pd.read_csv(input_csv_path,usecols=['random'])

    # 计算SVM得分
    scores_dict_low = seq_score_dict(svm_model, df['seq'].tolist())
    # scores_dict_high = seq_score_dict(svm_model, df2['random'].tolist())
    # 将得分添加到DataFrame
    df['predict'] = df['seq'].apply(lambda x: scores_dict_low.get(x))
    # df2['predict_random'] = df2['random'].apply(lambda x: scores_dict_high.get(x))


    # 保存新的CSV文件
    df.to_csv(output_csv_path, index=False)
    # df2.to_csv(output_csv_path, index=False)



if __name__ == "__main__":
    # svm_pene = '/home/lolo/lolo-bak/Code/AMP-GPT/AMP_SVM/model_pkl/svr_grid_outcomestrain_20210617_py38.pkl'
    svm_amp = '/home/leo/fightinglee/AMP-Projects/AMP/AMP-GPT/AMP_SVM/model_pkl/svr_model_20210115_01.pkl'
    svm_model = svm_amp

    # test_path = data_path['pepcvae_gen3']
    test_path = '/home/leo/fightinglee/AMP-Projects/Protein-Bert/data/E.coli-AMP-best.csv'
    # dict_seq_order = get_order(test_path)
    result_path = '/home/leo/fightinglee/AMP-Projects/Protein-Bert/data/E.coli-AMP-best_svm.csv'
    # result_path = test_path
    # seq_list = get_amp_list(test_path)
    # seq_and_final_score = seq_score_dict(svm_model,seq_list)


    # for i in sorted(seq_and_final_score.items(), key = lambda kv:(kv[1], kv[0]),reverse=True):
    #     with open(result_path,'a') as fw:
    #         fw.writelines(dict_seq_order[i[0]]+'\t'+i[0]+'\t'+str(i[1])+'\n')


    add_svm_scores_to_csv(test_path,svm_model,result_path)


