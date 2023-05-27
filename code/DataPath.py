data_path = {
    'penetratin':'data/penetratin/penetratin.txt',
    'penetratin_set':'data/penetratin/penetritin_high_activity_all.txt', 
    'penetratin_pom': '/home/leo/fightinglee/AMP-Projects/Protein-Bert/data/penetratin/penetratin_pom.txt',
    'penetratin_target': '/home/leo/fightinglee/AMP-Projects/Protein-Bert/data/penetratin/penetratin_targe.txt',
    'penetratin_mutants_1_general_model':'output/finetune/daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=random_pair,seed=6,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/reinforcement_proteins_package/RQIKIWFQNRRMKWKK_num_10.txt',
    'penetratin_mutants_2_general_model':'output/finetune/daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=random_pair,seed=6,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/Sample_polypeptide_package/daset=inter_test_RQIKIWFQNRRMKWKK,sample_times=20,batchsize=5,mask_ratio=0.3,max_pred=4,isPair=False,isRandom=False.txt',
    'penetratin_mutants_2_all_general_model':'output/finetune/daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=random_pair,seed=6,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/Sample_polypeptide_package/daset=inter_test_RQIKIWFQNRRMKWKK_all,sample_times=20,batchsize=5,mask_ratio=0.3,max_pred=4,isPair=False,isRandom=False.txt',
    'penetratin_mutants_2_general_model_merge':'output/finetune/daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=random_pair,seed=6,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/Sample_polypeptide_package/penetratin_mutants_2.txt',

    'penetratin_mutants_1_specific_model':'output/finetune/daset=random_pair,seed=6,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15_epoch_199/daset=random_chem_penetritin_high_activity_all_pair,seed=6,optimizer=Ranger,batch_size=32,epochs=200,lr=1e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/reinforcement_proteins_package/RQIKIWFQNRRMKWKK_num_10.txt',
    'penetratin_mutants_2_specific_model':'output/finetune/daset=random_pair,seed=6,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15_epoch_199/daset=random_chem_penetritin_high_activity_all_pair,seed=6,optimizer=Ranger,batch_size=32,epochs=200,lr=1e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/Sample_polypeptide_package/daset=inter_test_RQIKIWFQNRRMKWKK,sample_times=20,batchsize=5,mask_ratio=0.3,max_pred=4,isPair=False,isRandom=False.txt',
    'penetratin_mutants_2_all_specific_model':'output/finetune/daset=random_pair,seed=6,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15_epoch_199/daset=random_chem_penetritin_high_activity_all_pair,seed=6,optimizer=Ranger,batch_size=32,epochs=200,lr=1e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/Sample_polypeptide_package/daset=inter_test_RQIKIWFQNRRMKWKK_all,sample_times=20,batchsize=5,mask_ratio=0.3,max_pred=4,isPair=False,isRandom=False.txt',
    'penetratin_mutants_2_specific_model_merge':'output/finetune/daset=random_pair,seed=6,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15_epoch_199/daset=random_chem_penetritin_high_activity_all_pair,seed=6,optimizer=Ranger,batch_size=32,epochs=200,lr=1e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/Sample_polypeptide_package/penetratin_mutants_2.txt',
    'chem_gen1': '/home/leo/fightinglee/AMP-Projects/evolvX-0.1.0/seq-data/Experiment/G1.txt',
    'chem_gen2': '/home/leo/fightinglee/AMP-Projects/evolvX-0.1.0/seq-data/Experiment/G2.txt',
    'chem_gen3': '/home/leo/fightinglee/AMP-Projects/evolvX-0.1.0/seq-data/Experiment/G3.txt',
    'model_gen1':'/home/leo/fightinglee/AMP-Projects/Protein-Bert/output/finetune/daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=random_pair,seed=3,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/Sample_polypeptide_package/daset=inter_test_FFPIVGKLLSGLF,sample_times=20,batchsize=5,mask_ratio=0.3,max_pred=4,isPair=False,isRandom=False.txt',
    'model_gen2':'/home/leo/fightinglee/AMP-Projects/Protein-Bert/output/finetune/daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=random_pair,seed=3,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/Sample_polypeptide_package/daset=inter_test_FFPIVGKLLSGLF_best5,sample_times=20,batchsize=5,mask_ratio=0.3,max_pred=4,isPair=False,isRandom=False.txt',
    'model_gen3':'/home/leo/fightinglee/AMP-Projects/Protein-Bert/output/finetune/daset=uniport_all_60,seed=2,optimizer=Ranger,batch_size=256,epochs=350,lr=0.0007,weight_decay=0.0001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=30_epoch_349/daset=random_pair,seed=3,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/Sample_polypeptide_package/daset=inter_test_KFHLFKKILKGLF_best5_gen2,sample_times=20,batchsize=5,mask_ratio=0.3,max_pred=4,isPair=False,isRandom=False.txt',
    'random':'/home/leo/fightinglee/AMP-Projects/AMP/AMP-GPT/data/daset=random_pair,seed=6,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/reinforcement_proteins_package/FFPIVGKLLSGLF_num_10_random.txt_matrix_result.csv'
}


if __name__ == '__main__':
    import sys
    sys.path.append('.')
    from rwHelper import *

    a1, a2 = [lineTxtHelper(data_path[one]).readLines() for one in data_path.keys() if one in ('penetratin_mutants_2_general_model','penetratin_mutants_2_all_general_model')]
    a1, a2 = map(set, [a1,a2])
    c = a1 & a2
    d = a1 | a2
    c = c
    mutant_2_path = data_path['penetratin_mutants_2_general_model_merge']
    lineTxtHelper(mutant_2_path).writeLines(d)

    # from score_list import seq_score
    # svm_amp = '/home/lolo/lolo-bak/Code/AMP-GPT/AMP_SVM/model_pkl/svr_model_20210115_01.pkl'
    # svm_model = svm_amp
    # a1, a2 = [lineTxtHelper(data_path[one]).readLines() for one in data_path.keys() if one in ('penetratin_mutants_1_specific_model','penetratin_mutants_2_specific_model_merge')]
    # a1, a2 = map(set, [a1,a2])
    # d = a1 | a2
    # lines = list(zip(d, seq_score(svm_model, d)))
    # lines = sorted(lines, key=lambda one:one[1], reverse=True)
    # lines = [{'Seq':one, 'SVM value':score} for (one, score) in lines]
    # path = 'output/finetune/daset=random_pair,seed=6,optimizer=Ranger,batch_size=128,epochs=200,lr=6e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15_epoch_199/daset=random_chem_penetritin_high_activity_all_pair,seed=6,optimizer=Ranger,batch_size=32,epochs=200,lr=1e-05,weight_decay=0.001,lr_decay=True,lr_scheduler=WarmupLR,warmupEpochs=15/Sample_polypeptide_package/penetratin_mutants_new_round1.csv'
    # csvHelper(path).writeRows(lines)
    