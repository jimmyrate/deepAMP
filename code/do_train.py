import os
import shutil
import torch
import numpy as np
from dataPre import train_conf, model_conf
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import CyclicLR
from src.DataProcessTools import MaskCollate
from src.utils import save_checkpoint, configure_optimizers, configure_scheduler,checkDirs
from src.warmupScheduler import warmupLR
from alive_progress import alive_bar
from rwHelper import csvHelper, dicTxtHelper

def test(device, att_model, test_loader):
    with torch.no_grad():
        att_model.eval()
        losses = []
        # print("Running Testing")
        # with alive_bar(len(test_loader)) as bar:
        for batch_idx, (masked_seqs, masked_tokens, masked_pos) in enumerate(test_loader):
            # place data on the correct device
            masked_seqs = masked_seqs.to(device)
            masked_pos = masked_pos.to(device)
            masked_tokens = masked_tokens.to(device)

            # forward the model
            logits, loss = att_model(masked_seqs, masked_pos, masked_tokens)
            loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
            losses.append(loss.item())
                # report progress
                # bar(f"iter {batch_idx}: test loss {loss.item():.5f}")
        test_avg_loss = float(np.mean(losses))
    return test_avg_loss


def train():
    att_model = train_conf.model
    if train_conf.mode == 'finetune':
        att_model.load_state_dict(torch.load(train_conf.pretrained_model, map_location = lambda storage, loc:storage))
    train_dataset = train_conf.train_dataset
    test_dataset = train_conf.test_dataset
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
    if train_conf.mode == 'finetune':
        print(f'Pretrained model: "{train_conf.pretrained_model}"')
    print(train_conf.train_attr)    

    raw_model = att_model.module if hasattr(att_model, "module") else att_model
    # optimizer = raw_model.configure_optimizers(train_conf)
    optimizer = configure_optimizers(raw_model,train_conf.optimizer,train_conf.learning_rate,train_conf.weight_decay)
    # train_loader = DataLoader(train_dataset, shuffle=True, pin_memory=True,batch_size=train_conf.batch_size,num_workers=train_conf.num_workers, drop_last=True)
    pin_memory = False if d=='cpu' else True
    train_loader = DataLoader(train_dataset, 
            shuffle=True, pin_memory=pin_memory,batch_size=train_conf.batch_size,num_workers=train_conf.num_workers, drop_last=False,
            collate_fn=MaskCollate(train_dataset,mask=train_conf.mask, mask_ratio=train_conf.mask_ratio,max_pred=train_conf.max_pred))
    test_loader = DataLoader(test_dataset,
                shuffle=True, pin_memory=pin_memory,batch_size=train_conf.batch_size,num_workers=train_conf.num_workers, drop_last=False,
                collate_fn=MaskCollate(test_dataset,mask=train_conf.mask,mask_ratio=train_conf.mask_ratio,max_pred=train_conf.max_pred))
    scheduler = configure_scheduler(optimizer=optimizer,
                                train_conf=train_conf,
                                steps_per_epoch=len(train_loader))
    #######################
    #save train, model config
    #######################
    basicPath = checkDirs(train_conf.basicPath)
    dicTxtHelper(train_conf.modelconfigpath).writeDict(model_conf.getAttrs())
    dicTxtHelper(train_conf.trainconfigpath).writeDict(train_conf.getAttrs())

    # save pretrained model
    if train_conf.mode == 'finetune':
        pretrain_dir = train_conf.pretrained_model.rsplit('/', maxsplit=1)[0]
        basicPath = checkDirs(train_conf.basicPath)
        pretrain_save_path = f'{basicPath}epoch_-1'
        if os.path.exists(pretrain_save_path):
            shutil.rmtree(pretrain_save_path)
        shutil.copytree(pretrain_dir, pretrain_save_path)


    steps = 0
    for epoch in range(train_conf.max_epochs):
        att_model.train()
        losses = []
        print("Running EPOCH",epoch)
        with alive_bar(len(train_loader)) as bar:
            for batch_idx, (masked_seqs, masked_tokens, masked_pos) in enumerate(train_loader):
                steps += 1
                # place data on the correct device
                masked_seqs = masked_seqs.to(device)
                masked_pos = masked_pos.to(device)
                masked_tokens = masked_tokens.to(device)

                # forward the model
                logits, loss = att_model(masked_seqs, masked_pos, masked_tokens)
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())

                # backprop and update the parameters
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                att_model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(att_model.parameters(), train_conf.grad_norm_clip)
                optimizer.step()
                if train_conf.lr_decay:
                    scheduler.step()

                # report progress
                bar(f"epoch {epoch} iter {batch_idx}: train loss {loss.item():.5f}. lr {lr:e}")
                # if train_conf.doSave and steps%50==0:
                #     basicPath = checkDirs(train_conf.basicPath)
                #     basicPath = checkDirs(basicPath + f'steps_{steps}/')
                #     ckpt_path = basicPath + 'model.pkl'
                #     loss_path = basicPath + 'loss.txt'
                #     save_checkpoint(att_model, ckpt_path)
                #     dicTxtHelper(loss_path).writeDict({'loss':str(loss.item())})

        train_avg_loss = float(np.mean(losses))
        test_avg_loss = None
        title = '{:<10}  {:<30}' .format('epoch','train_avg_loss')
        rec = '{:<10}  {:<30}' .format(epoch,train_avg_loss)
        if train_conf.doTest:
            test_avg_loss = test(device,att_model, test_loader)
            title = '{:<10}  {:<30}  {:<30}' .format('epoch','train_avg_loss','test_avg_loss')
            rec = '{:<10}  {:<30}  {:<30}' .format(epoch,train_avg_loss, test_avg_loss)
        print(title)
        print(rec+'\n')

        if train_conf.doSave:
            basicPath = checkDirs(train_conf.basicPath)
            basicPath = checkDirs(basicPath + f'epoch_{epoch}/')
            ckpt_path = basicPath + 'model.pkl'
            loss_path = basicPath + 'loss.txt'

            save_checkpoint(att_model, ckpt_path)
            if train_conf.doTest:
                dicTxtHelper(loss_path).writeDict({'train_loss':str(train_avg_loss),'test_loss':str(test_avg_loss)})
            else:
                dicTxtHelper(loss_path).writeDict({'train_loss':str(train_avg_loss)})

if __name__ == "__main__":
    train()