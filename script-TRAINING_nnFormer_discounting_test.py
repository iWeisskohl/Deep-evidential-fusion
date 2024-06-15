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
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet_DS_fusion,DynUNet_ds_fusion_s
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
)
from monai.utils import set_determinism
from dataset.brats import get_datasets
import torch
from networks.nnFormer.nnFormer_discounting import nnFormer_ds_fusion_s

###############pre define#####@
from utils import AverageMeter, ProgressMeter, save_checkpoint, reload_ckpt_bis, \
    count_parameters, save_metrics, save_args_1, inference, dice_metric, \
    dice_metric_batch, generate_segmentations_monai

import calibration.stats as stats

from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss

###########cuda defination######

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
trained_path="best_metric_model_nnFormer_discounting_random_02.pth"
seed=1234
fold_number=2
batch_size=8
##################Set deterministic training for reproducibility##################
set_determinism(seed=0)

###########Define a new transform to convert brain tumor labels####

##############Setup transforms for training and validation################
#####To do: put transform in get_dataset.

full_train_dataset, l_val_dataset, bench_dataset = get_datasets(seed, fold_number=fold_number)
#train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=8, shuffle=True,
#                                               num_workers=4, pin_memory=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=4)
bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=4)

#print("Train dataset number of batch:", len(train_loader))
#print("Val dataset number of batch:", len(val_loader))
print("Bench Test dataset number of batch:", len(bench_loader))


###########Check data shape and visualize#########



########### Creat Model, Loss, optimizer##########
max_epochs = 1
val_interval = 1
VAL_AMP = True
model=nnFormer_ds_fusion_s().to(device)

#######Code for finetuning#######
'''
trained_model_path="best_metric_model.pth"
model_dict = model.state_dict()
pre_dict = torch.load(trained_model_path)
pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
model_dict.update(pre_dict)
model.load_state_dict(model_dict)
'''


##########Code for using the pre-trained model#############
model.load_state_dict(torch.load(trained_path))


#optimizer = torch.optim.Adam(model.parameters(), 1e-3)
#loss_function = monai.losses.DiceLoss(include_background=False,softmax=True,squared_pred=True,to_onehot_y=False)
'''
params = filter(lambda p: p.requires_grad, model.parameters())
for name, param in model.named_parameters():
    #print(name,param)
    if param.requires_grad==True:
        print(name)
'''

#loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, include_background=False,squared_pred=True, to_onehot_y=False, softmax=False)
#optimizer = torch.optim.Adam(model.parameters(), 1e-2)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=False, reduction="mean")
#dice_metric_c = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
dice_metric_batch_f = DiceMetric(include_background=True, reduction="mean_batch")


#post_trans = Compose(
#    [EnsureType(), Activations(Softmax=True), AsDiscrete(threshold=0.5)]
#)
post_trans = Compose(
    [EnsureType(), AsDiscrete(argmax=True, to_onehot=4)]
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
metric_values_ece = list()
metric_values_brier = list()
metric_values_nll = list()
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


for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():

            metric_sum = 0.0
            metric_sum_ece = 0.0
            metric_sum_brier = 0.0
            metric_sum_nll = 0.0
            metric_count = 0
            val_images = None
            val_labels = None
            val_outputs = None
            val_loader=bench_loader

            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                #val_images,val_labels=val_images.float(),val_labels.float()

                val_outputs=0


                val_outputs = inference(val_images,model)
                val_outputs_=val_outputs
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

                val_labels_ = torch.argmax(val_labels, dim=1, keepdim=True)
                val_labels_ = val_labels_.squeeze(0).cpu()
                val_labels_ = val_labels_.numpy()
                z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(val_labels_, axis=0) != 0)
                #z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(val_labels_, axis=0) != 0)
                zmin, ymin, xmin = [max(0, int(np.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)]
                zmax, ymax, xmax = [int(np.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)]
                gt = val_labels[:, :, zmin:zmax, ymin:ymax, xmin:xmax]

                onehot_targets = gt
                onehot_targets = onehot_targets.permute(1, 0, 2, 3, 4)
                onehot_targets = onehot_targets.reshape(4, -1)
                onehot_targets = onehot_targets.permute(1, 0)
                predictions = torch.stack(val_outputs)[:, :, zmin:zmax, ymin:ymax, xmin:xmax]

                brier = 0
                for i in range(4):
                    target = gt[:, i, ].reshape(-1)
                    predict = predictions[:, i].reshape(-1)
                    # if i>1:
                    brier = brier + brier_score_loss(target.cpu().numpy(), predict.cpu().numpy())

                predictions = predictions.permute(1, 0, 2, 3, 4)
                predictions = predictions.reshape(4, -1)
                predictions = predictions.permute(1, 0)
                predictions = predictions.cpu()
                onehot_targets = onehot_targets.cpu()
                predictions = predictions.numpy()
                onehot_targets = onehot_targets.numpy()
                ece = stats.ece(predictions, onehot_targets)
                nll_ = log_loss(onehot_targets, predictions)
                value = dice_metric(y_pred=val_outputs, y=val_labels)
                metric_count += len(value)

                metric_sum_ece += ece.item() * len(value)
                metric_sum_brier += brier.item() * len(value)
                metric_sum_nll += nll_.item() * len(value)


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

            metric_ece = metric_sum_ece / metric_count
            metric_values_ece.append(metric_ece)
            metric_brier = metric_sum_brier / metric_count
            metric_values_brier.append(metric_brier)
            metric_nll = metric_sum_nll / metric_count
            metric_values_nll.append(metric_nll)

            print("ece", metric_ece)
            print("brier", metric_brier)
            print("nll", metric_nll)

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

            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" ED: {metric_ed:.4f} ET: {metric_et:.4f} NRC: {metric_nrc:.4f}"
                f" ET: {metric_et_f:.4f} TC: {metric_tc_f:.4f} WT: {metric_wt_f:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}")

    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")

#######Plot the loss and metric###########
'''
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y, color="red")
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y, color="green")
plt.show()

plt.figure("train", (18, 6))
plt.subplot(1, 3, 1)
plt.title("Val Mean Dice ED")
x = [val_interval * (i + 1) for i in range(len(metric_values_ed))]
y = metric_values_ed
plt.xlabel("epoch")
plt.plot(x, y, color="blue")
plt.subplot(1, 3, 2)
plt.title("Val Mean Dice ET")
x = [val_interval * (i + 1) for i in range(len(metric_values_et))]
y = metric_values_et
plt.xlabel("epoch")
plt.plot(x, y, color="brown")
plt.subplot(1, 3, 3)
plt.title("Val Mean Dice NRC")
x = [val_interval * (i + 1) for i in range(len(metric_values_nrc))]
y = metric_values_nrc
plt.xlabel("epoch")
plt.plot(x, y, color="purple")
plt.show()
'''

########Check best model output with the input image and label#######

'''
model.load_state_dict(torch.load( "best_metric_model_nnunet_enn_fixds_s_0.pth"))
model.eval()
with torch.no_grad():
    # select one image to evaluate and visualize the model output
    val_input = l_val_dataset[6]["image"].unsqueeze(0).to(device)
    roi_size = (128, 128, 64)
    sw_batch_size = 4
    val_output = inference(val_input,model)
    val_output = post_trans(val_output[0])
    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"image channel {i}")
        plt.imshow(l_val_dataset[6]["image"][i, :, :, 70].detach().cpu(), cmap="gray")
    plt.show()
    # visualize the 3 channels label corresponding to this image
    plt.figure("label", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(l_val_dataset[6]["label"][i+1, :, :, 70].detach().cpu())
    plt.show()
    # visualize the 3 channels model output corresponding to this image
    plt.figure("output", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"output channel {i}")
        plt.imshow(val_output[i+1, :, :, 70].detach().cpu())
    plt.show()


'''