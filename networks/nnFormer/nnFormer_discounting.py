# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Sequence, Union
import torch
import torch.nn as nn
import monai
from monai.utils import alias, export
from monai.networks.blocks.dynunet_block import *
from torch.nn.parameter import Parameter
import numpy as np
from networks.nnFormer.nnFormer_s_ds_t1ce import nnFormer_s_ds_t1ce
from networks.nnFormer.nnFormer_s_ds_t1 import nnFormer_s_ds_t1
from networks.nnFormer.nnFormer_s_ds_flair import nnFormer_s_ds_flair
from networks.nnFormer.nnFormer_s_ds_t2 import nnFormer_s_ds_t2



class Fusion(nn.Module):
    def __init__(self, class_dim):
        super(Fusion, self).__init__()
        self.class_dim=class_dim

        self.alpha1 = Parameter(torch.Tensor(self.class_dim,1,1,1))
        self.alpha2 = Parameter(torch.Tensor(self.class_dim,1,1,1))
        self.alpha3 = Parameter(torch.Tensor(self.class_dim,1,1,1))
        self.alpha4 = Parameter(torch.Tensor(self.class_dim,1,1,1))
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.constant_(self.alpha1, 0)
        nn.init.constant_(self.alpha2, 0)
        nn.init.constant_(self.alpha3, 0)
        nn.init.constant_(self.alpha4, 0)


    def forward(self, input1,input2,input3,input4):
        [batch_size, class_dim, height, weight, depth] = input1.size()

        x1 = input1[:, :4,] +input1[:, 4,].unsqueeze(1)
        x2 = input2[:, :4,] +input2[:, 4,].unsqueeze(1)
        x3 = input3[:, :4,] +input3[:, 4,].unsqueeze(1)
        x4 = input4[:, :4,] +input4[:, 4,].unsqueeze(1)

        alpha1 = 1 / (1 + torch.exp(-self.alpha1))
        alpha2 = 1 / (1 + torch.exp(-self.alpha2))
        alpha3 = 1 / (1 + torch.exp(-self.alpha3))
        alpha4 = 1 / (1 + torch.exp(-self.alpha4))
        batch=torch.ones(batch_size, class_dim-1, height, weight, depth,device=input1.device)
        alpha1=batch*alpha1
        alpha2 = batch * alpha2
        alpha3 = batch * alpha3
        alpha4 = batch * alpha4


        a_x1 = alpha1 + (1 - alpha1) * x1
        a_x2 = alpha2 + (1 - alpha2) * x2
        a_x3 = alpha3 + (1 - alpha3) * x3
        a_x4 = alpha4 + (1 - alpha4) * x4

        pl = a_x1 * a_x2 * a_x3 * a_x4
        K = pl.sum(1)
        pl = (pl / (torch.ones(batch_size, class_dim-1, height, weight, depth, device=x1.device) * K.unsqueeze(1)))
        return pl

class nnFormer_ds_fusion_s(nn.Module):

    def __init__(self):
        super(nnFormer_ds_fusion_s, self).__init__()
        self.t1ce_modality = nnFormer_s_ds_t1ce(input_channels=1, num_classes=4)
        self.t1_modality = nnFormer_s_ds_t1(input_channels=1, num_classes=4)
        self.flair_modality = nnFormer_s_ds_flair(input_channels=1, num_classes=4)
        self.t2_modality = nnFormer_s_ds_t2(input_channels=1, num_classes=4)

        self.fusion = Fusion(4)


    def forward(self, x, train):
        x1 = x[:, 0,].unsqueeze(1)
        x2 = x[:, 1,].unsqueeze(1)
        x3 = x[:, 2,].unsqueeze(1)
        x4 = x[:, 3,].unsqueeze(1)
        x11 = self.t1ce_modality(x1)
        x22 = self.t1_modality(x2)
        x33 = self.flair_modality(x3)
        x44 = self.t2_modality(x4)

        x = self.fusion(x11,x22,x33,x44)
        
        if train==False:
           return x 
        else:
           return x,x11[:,:4, ],x22[:,:4,],x33[:,:4,],x44[:,:4,] 
