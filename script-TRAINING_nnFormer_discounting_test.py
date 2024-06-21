#!/usr/bin/env python
# coding: utf-8


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
from monai.transforms import AsDiscrete,Compose,EnsureTyped,EnsureType
from monai.utils import set_determinism
from dataset.brats import get_datasets
import torch
from networks.nnFormer.nnFormer_discounting import nnFormer_ds_fusion_s
from utils import  inference, dice_metric,dice_metric_batch
import calibration.stats as stats
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss

###########cuda defination######

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
print(device)
trained_path="./trained_model/best_metric_model_nnFormer_discounting.pth"
seed=1234
fold_number=0
######################Set deterministic training for reproducibility################
set_determinism(seed=0)

#########################################load data##################################
full_train_dataset, l_val_dataset, bench_dataset = get_datasets(seed, fold_number=fold_number)
bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=4)
print("Bench Test dataset number of batch:", len(bench_loader))


############################### Creat Model, Loss, optimizer##########################
model=nnFormer_ds_fusion_s().to(device)

##########################Code for using the pre-trained model########################
model.load_state_dict(torch.load(trained_path))


dice_metric = DiceMetric(include_background=False, reduction="mean")
dice_metric_batch = DiceMetric(include_background=False, reduction="mean_batch")
dice_metric_batch_f = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose( [EnsureType(), AsDiscrete(argmax=True, to_onehot=4)])

# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True


val_interval = 1
best_metric = -1
best_metric_epoch = -1
writer = SummaryWriter()
metric_values_ece = list()
metric_values_brier = list()
metric_values_nll = list()

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


for epoch in range(1):
    epoch_start = time.time()

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():

            metric_sum = 0.0
            metric_sum_ece = 0.0
            metric_sum_brier = 0.0
            metric_sum_nll = 0.0
            metric_count = 0
            val_loader=bench_loader

            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                #val_images,val_labels=val_images.float(),val_labels.float()

                val_outputs = inference(val_images,model)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                ######################################################################################################
                ############################segmentation uncertainty brier,nll, ece #################################
                ############################focus only on forground regions #################################
                ######################################################################################################
                val_labels_ = torch.argmax(val_labels, dim=1, keepdim=True)
                val_labels_ = val_labels_.squeeze(0).cpu()
                val_labels_ = val_labels_.numpy()
                z_indexes, y_indexes, x_indexes = np.nonzero(np.sum(val_labels_, axis=0) != 0)
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
                    brier = brier + brier_score_loss(target.cpu().numpy(), predict.cpu().numpy())

                predictions = predictions.permute(1, 0, 2, 3, 4)
                predictions = predictions.reshape(4, -1)
                predictions = predictions.permute(1, 0)
                predictions = predictions.cpu().numpy()
                onehot_targets = onehot_targets.cpu().numpy()

                ece = stats.ece(predictions, onehot_targets)
                nll_ = log_loss(onehot_targets, predictions)
                value = dice_metric(y_pred=val_outputs, y=val_labels)

                metric_count += len(value)
                metric_sum_ece += ece.item() * len(value)
                metric_sum_brier += brier.item() * len(value)
                metric_sum_nll += nll_.item() * len(value)


                ######################################################################################################
                ##########################segmentation accuracy: dice score ###########################################
                ######################################################################################################
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

                ############Code to transfer labels and predictions into three overlapping classes#####################

                ##### transfer for mask######
                patient_label = torch.argmax(val_labels.float(), dim=1, keepdim=True)
                patient_label = patient_label.squeeze(1).cpu()
                et = patient_label == 2
                tc = np.logical_or(patient_label == 2, patient_label == 3)
                wt = np.logical_or(tc, patient_label == 1)
                patient_label = np.stack([et, tc, wt], axis=1)
                patient_label = torch.from_numpy(patient_label)

                ########## transfer for prediction###########
                test = torch.stack(val_outputs)
                pred_label = torch.argmax(test, dim=1, keepdim=True)
                pred_label = pred_label.squeeze(1).cpu()
                et_ = pred_label == 2
                tc_ = np.logical_or(pred_label == 2, pred_label == 3)
                wt_ = np.logical_or(tc_, pred_label == 1)
                pred_label = np.stack([et_, tc_, wt_], axis=1)
                pred_label = torch.from_numpy(pred_label)
                ########## calculate dice score################
                dice_metric_batch_f(y_pred=pred_label, y=patient_label)

            metric_ece = metric_sum_ece / metric_count
            metric_values_ece.append(metric_ece)
            metric_brier = metric_sum_brier / metric_count
            metric_values_brier.append(metric_brier)
            metric_nll = metric_sum_nll / metric_count
            metric_values_nll.append(metric_nll)

            #####prediction for three single class ed, et, nrc ######
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
            #####prediction for three overlapping class et, tc, wt######
            metric_batch_f = dice_metric_batch_f.aggregate()
            metric_et_f = metric_batch_f[0].item()
            metric_values_et_f.append(metric_et_f)
            metric_tc_f = metric_batch_f[1].item()
            metric_values_tc_f.append(metric_tc_f)
            metric_wt_f = metric_batch_f[2].item()
            metric_values_wt_f.append(metric_wt_f)
            dice_metric_batch_f.reset()

            print("ece", metric_ece)
            print("brier", metric_brier)
            print("nll", metric_nll)
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" ED: {metric_ed:.4f} ET: {metric_et:.4f} NRC: {metric_nrc:.4f}"
                f" ET: {metric_et_f:.4f} TC: {metric_tc_f:.4f} WT: {metric_wt_f:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}")
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start
print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")