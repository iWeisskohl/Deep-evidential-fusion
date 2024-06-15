#!/usr/bin/env python
# coding: utf-8

# In[]:

###########################  IMPORTS   #############################################
from dice import EDiceLoss_Val
#import calibration
#import calibration.stats as stats
#import calibration.binning as binning
#import calibration.lenses as lenses


import os
import shutil
import tempfile
import time
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import monai
from torch.utils.tensorboard import SummaryWriter
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from networks.nnFormer.nnFormer_discounting import nnFormer_ds_fusion_s
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet_DS_fusion_s,DynUNet_ds_fusion_s
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    EnsureType,
)
from monai.utils import set_determinism
from dataset.brats import get_datasets
import torch


###############pre define#####@
from utils import AverageMeter, ProgressMeter, save_checkpoint, reload_ckpt_bis, \
    count_parameters, save_metrics, save_args_1, inference, dice_metric, \
    dice_metric_batch, generate_segmentations_monai


###########cuda defination######

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
seed=1234
fold_number=2
batch_size=4
saved_model="best_metric_model_nnFormer_discounting_random_02.pth"
##################Set deterministic training for reproducibility##################
set_determinism(seed=0)

###########Define a new transform to convert brain tumor labels####

##############Setup transforms for training and validation################
#####To do: put transform in get_dataset.

full_train_dataset, l_val_dataset, bench_dataset = get_datasets(seed, fold_number=fold_number)
train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=4)
bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=4)

print("Train dataset number of batch:", len(train_loader))
print("Val dataset number of batch:", len(val_loader))
print("Bench Test dataset number of batch:", len(bench_loader))


###########Check data shape and visualize#########



########### Creat Model, Loss, menoptimizer##########
max_epochs = 100
val_interval = 1
VAL_AMP = True
model=nnFormer_ds_fusion_s().to(device)

#######Code for finetuning#######
'''
trained_model_path="best_metric_model_nnFormer_discounting_01_.pth"
model_dict = model.state_dict()
pre_dict = torch.load(trained_model_path)
pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
model_dict.update(pre_dict)
model.load_state_dict(model_dict)
'''





#trained_model_path_1="/hpc/home/huang.l/research/TIP_miccai2022_extend_nnunet/brats_hl_medical-segmentation-master_t1ce_enn/best_metric_model_nnunet_01.pth"
#trained_model_path_1="/hpc/home/huang.l/research/TIP_miccai2022_extend_nnunet/brats_hl_medical-segmentation-master_t1ce_enn/best_metric_model_nnunet_01.pth"
trained_model_path_1="/hpc/home/huang.l/research/TIP_miccai2022_extend_nnunet/brats_hl_medical-segmentation-master_baseline_t1ce/best_metric_model_nnFormer_02.pth"
model_dict_1 = model.t1ce_modality.state_dict()
pre_dict_1 = torch.load(trained_model_path_1)
pre_dict_1 = {k: v for k, v in pre_dict_1.items() if k in model_dict_1}
model_dict_1.update(pre_dict_1)
model.t1ce_modality.load_state_dict(model_dict_1)

#trained_model_path_2="/hpc/home/huang.l/research/TIP_miccai2022_extend_nnunet/brats_hl_medical-segmentation-master_t1_enn/best_metric_model_nnunet_01.pth"
trained_model_path_2="/hpc/home/huang.l/research/TIP_miccai2022_extend_nnunet/brats_hl_medical-segmentation-master_baseline_t1/best_metric_model_nnFormer_02.pth"
model_dict_2 = model.t1_modality.state_dict()
pre_dict_2 = torch.load(trained_model_path_2)
pre_dict_2 = {k: v for k, v in pre_dict_2.items() if k in model_dict_2}
model_dict_2.update(pre_dict_2)
model.t1_modality.load_state_dict(model_dict_2)

#trained_model_path_3="/hpc/home/huang.l/research/TIP_miccai2022_extend_nnunet/brats_hl_medical-segmentation-master_flair_enn/best_metric_model_nnunet_01.pth"
trained_model_path_3="/hpc/home/huang.l/research/TIP_miccai2022_extend_nnunet/brats_hl_medical-segmentation-master_baseline_flair/best_metric_model_nnFormer_02.pth"
model_dict_3 = model.flair_modality.state_dict()
pre_dict_3 = torch.load(trained_model_path_3)
pre_dict_3 = {k: v for k, v in pre_dict_3.items() if k in model_dict_3}
model_dict_3.update(pre_dict_3)
model.flair_modality.load_state_dict(model_dict_3)


#trained_model_path_4="/hpc/home/huang.l/research/TIP_miccai2022_extend_nnunet/brats_hl_medical-segmentation-master_t2_enn/best_metric_model_nnunet_01.pth"
trained_model_path_4="/hpc/home/huang.l/research/TIP_miccai2022_extend_nnunet/brats_hl_medical-segmentation-master_baseline_t2/best_metric_model_nnFormer_02.pth"
model_dict_4 = model.t2_modality.state_dict()
pre_dict_4 = torch.load(trained_model_path_4)
pre_dict_4 = {k: v for k, v in pre_dict_4.items() if k in model_dict_4}
model_dict_4.update(pre_dict_4)
model.t2_modality.load_state_dict(model_dict_4)


#import torchinfo
#summary = torchinfo.summary(model, input_size=(1,4,128,128,128), device='cpu')

##########Code for using the pre-trained model#############
#model.load_state_dict(torch.load(saved_model))


#optimizer = torch.optim.Adam(model.parameters(), 1e-3)
#loss_function = monai.losses.DiceLoss(include_background=False,so

params = filter(lambda p: p.requires_grad, model.parameters())
for name, param in model.named_parameters():
    #print(name,param)
    if param.requires_grad==True:
        print(name)

#feature_params = model.features.parameters()
#ds_params = model.ds.parameters()

#optimizer = optim.Adam([{'params': feature_params, 'lr': 0.001},{'params': classifier_params, 'lr': 0.01}])

loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, include_background=False,squared_pred=True, to_onehot_y=False, softmax=False)
#optimizer = torch.optim.Adam(model.parameters(), 1e-2)
optimizer = torch.optim.Adam(params, 1e-2)    ####be careful here, in the previous version,  only "params" are used
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=False, reduction="mean")
#dice_metric_c = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
dice_metric_batch_f = DiceMetric(include_background=True, reduction="mean_batch")


#post_trans = Compose(
#    [EnsureType(), Activations(Softmax=True), AsDiscrete(threshold=0.5)]
#)
post_trans = Compose(
    [EnsureType(), Activations(softmax=False), AsDiscrete(argmax=True, to_onehot=4)]
)
# define inference method



# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

########Code for transfer 4 class data into 3 class###########
val_interval = 1
best_metric = -1
best_metric_epoch = -1
writer = SummaryWriter()
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=4)
post_label = AsDiscrete(to_onehot=True, n_classes=4)


best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_ed = []
metric_values_et = []
metric_values_nrc = []

metric_values_et_f = []
metric_values_tc_f = []
metric_values_wt_f = []
total_start = time.time()


patience = 10  # Number of epochs to wait for improvement
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
    #lr_scheduler.step()
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
                #val_images,val_labels=val_images.float(),val_labels.float()

                val_outputs = inference(val_images,model)
                val_outputs_=val_outputs
                #print(val_outputs.shape)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]


                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

                ############Code for transfer to three class modality###########

                ##### transfer for mask######
                patient_label = torch.argmax(val_labels.float(), dim=1, keepdim=True)
                patient_label = patient_label.squeeze(1).cpu()
                et = patient_label == 2
                tc = np.logical_or(patient_label == 2, patient_label == 3)
                wt = np.logical_or(tc, patient_label == 1)
                patient_label = np.stack([et, tc, wt], axis=1)
                patient_label = torch.from_numpy(patient_label)

                ##### transfer for prediction######
                test = torch.stack(val_outputs)
                pred_label = torch.argmax(test, dim=1, keepdim=True)
                pred_label = pred_label.squeeze(1).cpu()
                et_ = pred_label == 2
                tc_ = np.logical_or(pred_label == 2, pred_label == 3)
                wt_ = np.logical_or(tc_, pred_label == 1)
                pred_label = np.stack([et_, tc_, wt_], axis=1)
                pred_label = torch.from_numpy(pred_label)

                dice_metric_batch_f(y_pred=pred_label, y=patient_label)


            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_ed = metric_batch[0].item()
            metric_values_ed.append(metric_ed)
            metric_et = metric_batch[1].item()
            metric_values_et.append(metric_et)
            metric_nrc = metric_batch[2].item()
            metric_values_nrc.append(metric_nrc)
            dice_metric.reset()
            dice_metric_batch.reset()

            metric_batch_f = dice_metric_batch_f.aggregate()
            metric_et_f = metric_batch_f[0].item()
            metric_values_et_f.append(metric_et_f)
            metric_tc_f = metric_batch_f[1].item()
            metric_values_tc_f.append(metric_tc_f)
            metric_wt_f = metric_batch_f[2].item()
            metric_values_wt_f.append(metric_wt_f)
            dice_metric_batch_f.reset()


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
                f" ED: {metric_ed:.4f} ET: {metric_et:.4f} NRC: {metric_nrc:.4f}"
                f" ET: {metric_et_f:.4f} TC: {metric_tc_f:.4f} WT: {metric_wt_f:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}")

    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")

        # Early stopping
    if len(metric_values) >= patience:
        recent_metrics = metric_values[-patience:]
        if all(metric <= best_metric * (1 - min_improvement) for metric in recent_metrics):
            print(f"No improvement in the last {patience} epochs. Stopping training.")
            break

total_time = time.time() - total_start
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
