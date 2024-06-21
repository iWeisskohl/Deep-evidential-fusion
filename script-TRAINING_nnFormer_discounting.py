#!/usr/bin/env python
# coding: utf-8



###########################  IMPORTS   #############################################

import time
import numpy as np
import monai
from torch.utils.tensorboard import SummaryWriter
from monai.data import DataLoader, decollate_batch
from networks.nnFormer.nnFormer_discounting import nnFormer_ds_fusion_s
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations,AsDiscrete,Compose,EnsureType
from monai.utils import set_determinism
from dataset.brats import get_datasets
import torch


###############pre define#####@
from utils import  inference, dice_metric, dice_metric_batch


###########cuda defination######

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(device)
seed=1234
fold_number=0
batch_size=8
max_epochs = 100
val_interval = 1
saved_model="./trained_model/best_metric_model_nnFormer_discounting.pth"

##################Set deterministic training for reproducibility##################
set_determinism(seed=0)


##############load dataset ################

full_train_dataset, l_val_dataset, bench_dataset = get_datasets(seed, fold_number=fold_number)
train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=0, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=0)
bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=0)

print("Train dataset number of batch:", len(train_loader))
print("Val dataset number of batch:", len(val_loader))
print("Bench Test dataset number of batch:", len(bench_loader))



########### Creat Model##########
model=nnFormer_ds_fusion_s().to(device)


trained_model_path_1="./Pretrained_model/t1Gd/best_metric_model_nnFormer.pth"
model_dict_1 = model.t1ce_modality.state_dict()
pre_dict_1 = torch.load(trained_model_path_1)
pre_dict_1 = {k: v for k, v in pre_dict_1.items() if k in model_dict_1}
model_dict_1.update(pre_dict_1)
model.t1ce_modality.load_state_dict(model_dict_1)

trained_model_path_2="./Pretrained_model/t1/best_metric_model_nnFormer.pth"
model_dict_2 = model.t1_modality.state_dict()
pre_dict_2 = torch.load(trained_model_path_2)
pre_dict_2 = {k: v for k, v in pre_dict_2.items() if k in model_dict_2}
model_dict_2.update(pre_dict_2)
model.t1_modality.load_state_dict(model_dict_2)

trained_model_path_3="./Pretrained_model/Flair/best_metric_model_nnFormer.pth"
model_dict_3 = model.flair_modality.state_dict()
pre_dict_3 = torch.load(trained_model_path_3)
pre_dict_3 = {k: v for k, v in pre_dict_3.items() if k in model_dict_3}
model_dict_3.update(pre_dict_3)
model.flair_modality.load_state_dict(model_dict_3)


trained_model_path_4="./Pretrained_model/t2/best_metric_model_nnFormer.pth"
model_dict_4 = model.t2_modality.state_dict()
pre_dict_4 = torch.load(trained_model_path_4)
pre_dict_4 = {k: v for k, v in pre_dict_4.items() if k in model_dict_4}
model_dict_4.update(pre_dict_4)
model.t2_modality.load_state_dict(model_dict_4)



########### Creat Loss, menoptimizer##########
params = filter(lambda p: p.requires_grad, model.parameters())
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, include_background=False,squared_pred=True, to_onehot_y=False, softmax=False)
optimizer = torch.optim.Adam(params, 1e-2)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
dice_metric = DiceMetric(include_background=False, reduction="mean")

########### post-processing##########
post_trans = Compose([EnsureType(), Activations(softmax=False), AsDiscrete(argmax=True, to_onehot=4)])



# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

writer = SummaryWriter()
best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
total_start = time.time()

patience = 10 # Number of epochs to wait for improvement
min_improvement = 0.01  # Minimum required improvement to consider as progress


for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")

    model.train()
    epoch_loss = 0
    step = 0
    

    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        #inputs,labels=inputs.float(),labels.float()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs,output1,output2,output3,output4 = model(inputs,True)
            loss = loss_function(outputs, labels)+loss_function(output1, labels)+loss_function(output2, labels)+loss_function(output3, labels)+loss_function(output4, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        epoch_len = len(full_train_dataset) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():

            metric_sum = 0.0
            metric_count = 0

            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = inference(val_images,model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)


            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            dice_metric.reset()

            if metric > best_metric+ min_improvement:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(model.state_dict(), saved_model)
                print("saved new best metric model")
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                   print("Early stopping triggered.")
                   break
            
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}")

    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")


total_time = time.time() - total_start
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
