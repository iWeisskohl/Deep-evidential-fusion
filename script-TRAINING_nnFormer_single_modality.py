#!/usr/bin/env python
# coding: utf-8

# In[]:

###########################  IMPORTS   #############################################



import time
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import monai
from torch.utils.tensorboard import SummaryWriter
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)
from monai.utils import set_determinism
from dataset.brats import get_datasets
import torch
from einops import rearrange
from copy import deepcopy
from torch import nn
import torch
import numpy as np
from networks.nnFormer.nnFormer_single import nnFormer_s
import torch.nn.functional

from utils import inference, dice_metric, \
    dice_metric_batch


###########cuda defination######

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(device)


##################Set deterministic training for reproducibility##################
set_determinism(seed=0)


saved_model="best_metric_model_nnFormer.pth"
seed=1234
fold_number=0
batch_size=4
max_epochs = 50
val_interval = 50

full_train_dataset, l_val_dataset, bench_dataset = get_datasets(seed, fold_number=fold_number)
train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=4)
bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=4)

print("Train dataset number of batch:", len(train_loader))
print("Val dataset number of batch:", len(val_loader))
print("Bench Test dataset number of batch:", len(bench_loader))


########### Creat Model, Loss, optimizer##########

VAL_AMP = True
model =nnFormer_s(input_channels=1, num_classes=4).to(device)


loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, include_background=False,squared_pred=True, to_onehot_y=False, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=False, reduction="mean")
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
dice_metric_batch_f = DiceMetric(include_background=True, reduction="mean_batch")



post_trans = Compose(
    [EnsureType(), Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=4)]
)



# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
writer = SummaryWriter()
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=4)
post_label = AsDiscrete(to_onehot=True, n_classes=4)

best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []

total_start = time.time()


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
        inputs,labels=inputs.float(),labels.float()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs[:,0,].unsqueeze(1),False) ####Change the index here to input differnet modalities 
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        epoch_len = len(full_train_dataset) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():

            metric_sum = 0.0
            metric_count = 0
            val_images = None
            val_labels = None
            val_outputs = None


            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_images,val_labels=val_images.float(),val_labels.float()

                val_outputs = inference(val_images,model)
                val_outputs_=val_outputs
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]


                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)

            dice_metric.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(model.state_dict(), saved_model)
                print("saved new best metric model")
                        
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" at epoch: {best_metric_epoch}")

            


    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
