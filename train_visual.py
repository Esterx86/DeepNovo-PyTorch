import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
#from torch.utils.data import random_split, DataLoader
#from torch.optim import Adam, SGD
#from torch.optim.lr_scheduler import LambdaLR,StepLR
from torch.utils.tensorboard import SummaryWriter
#from torch.nn.utils.rnn import pack_padded_sequence

import config
import model
#from loadspec import load_data
#from preprocess import RawDataset,SpecDataset, BucketSampler, train_preprocess, test_preprocess
from visualization import masked_heatmap

writer = SummaryWriter('./log')
#start=datetime.now()
#train_data=load_data("data/cross_valid/cross.cat.mgf.train.repeat")
#val_data=load_data("data/cross_valid/cross.cat.mgf.valid.repeat")
#train_data = load_data("data/human_high/peaks.db.mgf.train.unique1")
#val_data = load_data("data/human_high/peaks.db.mgf.valid.unique1")
#end=datetime.now()
#print(f'data loaded. Time: {end-start}')

#train_set = RawDataset(train_data)
#val_set = RawDataset(val_data)

#train_sampler = BucketSampler(data_source=train_set,
#                              batch_size=config.BATCH_SIZE, buckets=config._buckets)
#train_loader=DataLoader(train_set,batch_sampler=train_sampler,collate_fn=train_preprocess,num_workers=6)
#val_sampler = BucketSampler(data_source=val_set,
#                              batch_size=config.BATCH_SIZE, buckets=config._buckets,shuffle=False)
#val_loader=DataLoader(val_set,batch_sampler=val_sampler,collate_fn=train_preprocess,num_workers=6)

#train_model=model.Model(direction=2).to(device=config.device)
#writer.add_graph(train_model)
#checkpoint=torch.load('checkpoint_human.pth')
#train_model.load_state_dict(checkpoint["model"])
masses = torch.tensor(config.masses_np, dtype=config.DTYPE, device=config.device)
#lr_scheduler=StepLR(train_model.opt,step_size=20,gamma=0.5)

def train_loop(model,data_loader,direction=2):
    total_loss=[]
    tp=0
    pp=0
    total_true=0
    total_pre=0

    for (spectrum_batch,
        fragments_forward_batch,
        fragments_backward_batch,
        target_forward_batch,
        target_backward_batch,
        weight_forward_batch,
        weight_backward_batch) in data_loader:

        spectrum=spectrum_batch.to(dtype=config.DTYPE,device=config.device)
        fragments_forward=fragments_forward_batch.to(dtype=config.DTYPE,device=config.device)
        fragments_backward=fragments_backward_batch.to(dtype=config.DTYPE,device=config.device)
        target_forward=target_forward_batch.to(dtype=torch.long,device=config.device)
        target_backward=target_backward_batch.to(dtype=torch.long,device=config.device)
        weight_forward=weight_forward_batch.to(dtype=config.DTYPE,device=config.device)
        weight_backward=weight_backward_batch.to(dtype=config.DTYPE,device=config.device)

        (loss_batch,
         tp_batch, pp_batch,
         total_true_batch, total_pre_batch,
         outputs_forward_batch, outputs_backward_batch) = model.train_step(
            spectrum=spectrum,
            fragments_forward=fragments_forward,
            fragments_backward=fragments_backward,
            sequence_forward=target_forward,
            sequence_backward=target_backward,
            target_weights_forward=weight_forward,
            target_weights_backward=weight_backward,
            direction=direction)
        
        total_loss.append(loss_batch)
        tp+=tp_batch
        pp+=pp_batch
        total_true+=total_true_batch
        total_pre+=total_pre_batch

    total_loss_mean = np.mean(total_loss)
    precision=pp/total_pre
    recall=tp/total_true

    return total_loss_mean, precision,recall

def val_loop(model,data_loader, direction=2, visualize=[]):
    total_loss=[]
    tp=0
    pp=0
    total_true=0
    total_pre=0
    figs=[]
    
    for i, (spectrum_batch,
        fragments_forward_batch,
        fragments_backward_batch,
        target_forward_batch,
        target_backward_batch,
        weight_forward_batch,
        weight_backward_batch) in enumerate(data_loader): # 这个可以弄成完全不随机嘛

        spectrum=spectrum_batch.to(dtype=config.DTYPE,device=config.device)
        fragments_forward=fragments_forward_batch.to(dtype=config.DTYPE,device=config.device)
        fragments_backward=fragments_backward_batch.to(dtype=config.DTYPE,device=config.device)
        target_forward=target_forward_batch.to(dtype=torch.long,device=config.device)
        target_backward=target_backward_batch.to(dtype=torch.long,device=config.device)
        weight_forward=weight_forward_batch.to(dtype=config.DTYPE,device=config.device)
        weight_backward=weight_backward_batch.to(dtype=config.DTYPE,device=config.device)

        (loss_batch,
         tp_batch, pp_batch,
         total_true_batch, total_pre_batch,
         outputs_forward_batch, outputs_backward_batch) = model.val_step(
            spectrum=spectrum,
            fragments_forward=fragments_forward,
            fragments_backward=fragments_backward,
            sequence_forward=target_forward,
            sequence_backward=target_backward,
            target_weights_forward=weight_forward,
            target_weights_backward=weight_backward,
            direction=direction)
        
        if i in visualize:
            row = 2
            fig, axes=plt.subplots(row,1,figsize=(8,2.5),dpi=300)
            plot_heatmap(axes[0],outputs_forward_batch,target_forward,5,step=1)
            plot_heatmap(axes[1],outputs_backward_batch,target_backward,5,step=-1)
            figs.append(fig)
            plt.close ("all")
            
        total_loss.append(loss_batch)
        tp+=tp_batch
        pp+=pp_batch
        total_true+=total_true_batch
        total_pre+=total_pre_batch

    total_loss_mean = np.mean(total_loss)
    precision=pp/total_pre
    recall=tp/total_true

    return total_loss_mean, precision,recall, figs

def test_loop(model,data_loader,teacher_forcing_ratio=1.0):
    total_loss=[]
    hits=0
    totals=0
    for (spectrum_batch,
         spectrum_forward_batch,
         spectrum_backward_batch,
         #fragments_forward_batch,
         #fragments_backward_batch,
         target_backward_batch,
         target_forward_batch,
         #weight_forward_batch,
         #weight_backward_batch,
         neutral_mass_batch) in data_loader:

        spectrum=spectrum_batch.to(dtype=config.DTYPE,device=config.device)
        spectrum_forward=spectrum_forward_batch.to(dtype=config.DTYPE,device=config.device)
        spectrum_backward=spectrum_backward_batch.to(dtype=config.DTYPE,device=config.device)
        #fragments_forward=fragments_forward_batch.to(dtype=config.DTYPE,device=config.device)
        #fragments_backward=fragments_backward_batch.to(dtype=config.DTYPE,device=config.device)
        target_forward=target_forward_batch.to(dtype=torch.long,device=config.device)
        target_backward=target_backward_batch.to(dtype=torch.long,device=config.device)
        #weight_forward=weight_forward_batch.to(dtype=config.DTYPE,device=config.device)
        #weight_backward=weight_backward_batch.to(dtype=config.DTYPE,device=config.device)
        neutral_mass=neutral_mass_batch.to(dtype=config.DTYPE,device=config.device)

        loss_batch,hit_batch,total_batch=model.val_loop(
            spectrum=spectrum,
            spectrum_forward=spectrum_forward,
            spectrum_backward=spectrum_backward,
            #fragments_forward=fragments_forward,
            #fragments_backward=fragments_backward,
            sequence_forward=target_forward,
            sequence_backward=target_backward,
            #target_weights_forward=weight_forward,
            #target_weights_backward=weight_backward,
            pepmass=neutral_mass,
            direction=2,
            teacher_forcing_ratio=teacher_forcing_ratio)
        
        total_loss.append(loss_batch)
        hits += hit_batch
        totals += total_batch

    total_loss_mean = np.mean(total_loss)
    accuracy = hits/totals
    # #print(f'mean loss: {total_loss_mean}')
    return total_loss_mean, accuracy

def plot_heatmap(ax,outputs,targets,n,step=1):
    if isinstance(outputs, torch.Tensor)==False:
        return ax
    output=outputs[:,:n].T.cpu().numpy()[:,::step] # (samples, L-1)
    target=targets[1:,:n].T.cpu().numpy()[:,::step] # (samples, L-1)
    
    ax=masked_heatmap(output,target,ax)
    #writer.add_figure("Sequence", fig, global_step=None)
    return ax

def save_checkpoint(model,optimizer,epoch,save_path,loss=None,accuracy=None):
    checkpoint={
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss":loss,
        "accuracy":accuracy,
    }
    torch.save(checkpoint,save_path)

def train(
    model,
    optimizer,
    train_loader,
    val_loader,
    n_iters=100,
    save_path=None,
    lr_scheduler=None,
    resume=None,
    direction=2):

    train_losses=[]
    val_losses=[]
    best_val_loss = np.inf
    best_val_pre = 0.0
    best_val_rec = 0.0
    best_val_epoch = 0

    current_epoch = 0
    if resume != None:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        current_epoch = checkpoint["epoch"]

    for epoch in tqdm(range(current_epoch, current_epoch+n_iters)):
        start = datetime.now()
        train_loss, train_pre, train_rec = train_loop(model, train_loader, direction)
        val_loss, val_pre, val_rec, val_figs = val_loop(model, val_loader, direction, visualize=[10, 30, 50])
        # val_loss, val_acc = val_loop(train_model, val_loader)
        if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            lr_scheduler.step()
        end = datetime.now()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_epoch = epoch
            best_val_loss = val_loss
            best_val_pre = val_pre
            best_val_rec = val_rec
            if save_path:
                save_checkpoint(model=model, optimizer=optimizer,
                                epoch=epoch, save_path=save_path)
            # #torch.save(train_model.state_dict(), f'./best_epoch_2.pth')
            # torch.save(checkpoint,f'./checkpoint.pth')

        writer.add_scalars(
            "Loss", {"Train loss": train_loss, "Validation loss": val_loss}, epoch)
        writer.add_scalars(
            "Precision", {"Train precision": train_pre, "Validation precision": val_pre}, epoch)
        writer.add_scalars(
            "Recall", {"Train recall": train_rec, "Validation recall": val_rec}, epoch)
        writer.add_figure("Sequence", val_figs, global_step=epoch, close=True)
        #writer.add_scalar("Loss", train_loss, epoch)
        #writer.add_scalar("Train Accuracy", train_acc, epoch)
        print(
            f'epoch: {epoch+1}\ttime: {end-start}\n\
            train_loss: {train_loss:.4f}\ttrain_precision: {train_pre:.4f}\ttrain_recall: {train_rec:.4f}\n\
            val_loss: {val_loss:.4f}\tval_precision: {val_pre:.4f}\tval_recall:{val_rec:.4f}\n\
            best_epoch: {best_val_epoch}\tloss: {best_val_loss:.4f}\tpre: {best_val_pre:.4f}\trec: {best_val_rec:.4f}')

        if epoch%20==0:
            torch.save(model.state_dict(), f'./ckpt/human_{epoch}.pth')
    return train_losses,val_losses

#writer.add_text()

