import os
from src.utils import checkDirs
from rwHelper import dicTxtHelper
class BertConfig:
    n_layers = 6
    n_heads = 12
    n_embd = 35
    d_model = 768
    d_ff = 768*4 # 4*d_model, FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V


    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)
    
    def getAttrs(self):
        return self.__dict__


class TrainerConfig:
    # optimization parameters
    daset = 'None'
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_epochs=[2]
    total_epochs=[max_epochs]
    init_lr=[learning_rate*1e-2]
    max_lr=[learning_rate]
    final_lr=[learning_rate*1e-3]
    optimizer = 'Ranger'
    mode = 'finetune'

    # bert train params
    mask=True
    mask_ratio=0.3
    max_pred=20
    
    # checkpoint settings
    pretrained_model = None
    doSave = True
    doTest = True
    num_workers = 0 # for DataLoader
    pretrained_model=None


    # random seed
    seed = 1

    # 
    experience_replay=False

    def __init__(self, **kwargs):
        self.alterAttr(**kwargs)

    
    def alterAttr(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        if not isinstance(self.warmup_epochs, list):
            self.warmup_epochs = [self.warmup_epochs]
        self.init_lr=[self.learning_rate*1e-2]
        self.max_lr=[self.learning_rate]
        self.final_lr=[self.learning_rate*1e-2]
        self.total_epochs=[self.max_epochs]

        if not self.lr_decay:
            # self.train_attr = ''.join(['daset=',str(self.daset),',optimizer=',str(self.optimizer),',batch_size=',str(self.batch_size),',epochs=',str(self.max_epochs),',lr=',str(self.learning_rate),',weight_decay=',str(self.weight_decay),',lr_decay=',str(self.lr_decay)])
            self.train_attr = ''.join(['daset=',str(self.daset),',seed=',str(self.seed),',optimizer=',str(self.optimizer),',batch_size=',str(self.batch_size),',epochs=',str(self.max_epochs),',lr=',str(self.learning_rate),',weight_decay=',str(self.weight_decay),',lr_decay=',str(self.lr_decay)])
        else:
            self.train_attr = ''.join(['daset=',str(self.daset),',seed=',str(self.seed),',optimizer=',str(self.optimizer),',batch_size=',str(self.batch_size),',epochs=',str(self.max_epochs),',lr=',str(self.learning_rate),',weight_decay=',str(self.weight_decay),',lr_decay=',str(self.lr_decay)
                            ,',lr_scheduler=',str(self.lr_scheduler),',warmupEpochs=',str(self.warmup_epochs[0])])

        self.basicPath = f'./output/{self.mode}/{self.train_attr}/'
        if self.mode != 'pretrain' and self.pretrained_model is not None:
            self.pretrain_model_path_name = '_'.join(self.pretrained_model.split('/')[-3:-1])
            self.basicPath = f'./output/{self.mode}/{self.pretrain_model_path_name}/{self.train_attr}/'
        self.modelconfigpath = f'{self.basicPath}model_config.txt'
        self.trainconfigpath = f'{self.basicPath}train_config.txt'
        self.generateconfigpath = f'{self.basicPath}generate_config.txt'
        

    def getAttrs(self):
        d = self.__dict__
        try:
            del d['model']
            del d['train_dataset']
            del d['test_dataset']
        except:
            pass
        return d


class ReinforceConfig:
    reinforce_times=3
    mask=False
    mask_ratio=0.3
    max_pred=20
    batch_size = 10
    isRandom = False
    src_daset = None
    optimize_amp = 'FFPIVGKLLSGLL'
    config_file = f'{optimize_amp}_config.txt'
    def __init__(self, **kwargs):
        self.alterAttr(**kwargs)
    
    def alterAttr(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.config_file = f'{self.optimize_amp}_config.txt'
    
    def getAttrs(self):
        d = self.__dict__
        return d
    
class makeSampleConfig:
    daset = 'None'
    dataset = None
    isRandom = False
    ForceRandom = False
    isPair = False
    sample_times = 10
    batch_size = 64
    sample_model = None
    protein_path=None

    mask=True
    mask_ratio=0.3
    max_pred=20

    sample_outside_dir = 'output/Sample_polypeptide/'
    sample_model_path_name = None
    outsidePath = None
    outside_record = True

    def __init__(self, **kwargs):
        self.alterAttr(**kwargs)

    
    def alterAttr(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)
        if self.sample_model is not None:
            self.sample_model_path_name = '_'.join(self.sample_model.split('/')[-4:-1])
            self.outsidePath = f'{self.sample_outside_dir}{self.sample_model_path_name}/'

    def getAttrs(self):
        d = self.__dict__
        try:
            del d['model']
            del d['original_dataset']
        except:
            pass
        return d


