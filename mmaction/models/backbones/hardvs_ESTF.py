# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import math
from torch.nn.init import kaiming_normal_, constant_
from .hardutils import predict_flow, crop_like, conv_s, conv, deconv, conv_s_p, conv_ac 

import matplotlib.pyplot as plt 

import torch.nn.functional as F
import numpy as np 
import cv2 


from collections import OrderedDict

import warnings 
warnings.filterwarnings("ignore")
import pdb 
import matplotlib.pyplot as plt

from ...utils import get_root_logger
from ..builder import BACKBONES
from mmcv.utils import _BatchNorm
from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init
from mmcv.runner import BaseModule

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from mmaction.models.backbones import torchvision_resnet 
import random


####################################
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
  
class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, attn_drop=0., proj_drop=0.,to_device="cuda:2"):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


@BACKBONES.register_module()
class ESTF(nn.Module):
    expansion = 1
    # default_hyper_params = dict(pretrain_model_path="", crop_pad=4, pruned=True,)
    def __init__(self, pretrained=None, batchNorm=True, output_layers=None, init_std=0.05,
                dim=24, clip_len=9, num_heads=6, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None, drop_path=0.,to_device="cuda:2"):  
        super(ESTF, self).__init__()
        self.batchNorm = batchNorm

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)
        temporal_dim=dim
        spatial_dim = dim
        self.snn_fc = nn.Sequential(
            # nn.Linear(8192, 4096),
            nn.Linear(6144, 3072),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.5),
            nn.Linear(3072, 3072),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.5),
            # nn.Linear(4096, 280)
            # nn.Linear(4096, 10)
        )
        self.device = to_device
        self.norm1 = nn.LayerNorm(temporal_dim)
        self.norm2 = nn.LayerNorm(spatial_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm5 = nn.LayerNorm(dim)
        self.norm6 = nn.LayerNorm(dim)
        self.norm7 = nn.LayerNorm(dim)
        self.norm8 = nn.LayerNorm(dim)
        self.norm9 = nn.LayerNorm(dim)
        self.norm10 = nn.LayerNorm(dim)




      
        self.attn1 = Attention(temporal_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,to_device=self.device)
        self.attn2 = Attention(spatial_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,to_device=self.device)
        self.attn3 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,to_device=self.device)
        self.attn4 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,to_device=self.device)
        self.attn5 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,to_device=self.device)

        self.ls1 = LayerScale(temporal_dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(temporal_dim, init_values=init_values) if init_values else nn.Identity()
        self.ls3 = LayerScale(spatial_dim, init_values=init_values) if init_values else nn.Identity()
        self.ls4 = LayerScale(spatial_dim, init_values=init_values) if init_values else nn.Identity()
        self.ls5 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls6 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls7 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls8 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls9 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls10 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp1 = Mlp(in_features=temporal_dim, hidden_features=int(temporal_dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        self.mlp2 = Mlp(in_features=spatial_dim, hidden_features=int(spatial_dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        self.mlp3 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        self.mlp4 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        self.mlp5 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)


        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        embed_dim=dim
        drop_rate = 0.
        self.clip_len = clip_len
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.clip_len, temporal_dim))
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, 16, spatial_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.resnet18_feature_extractor = torchvision_resnet.resnet18(pretrained = True)
        self.res_out_temporal_conv2d = conv_s_p(self.batchNorm, in_planes=512, out_planes=16, kernel_size=4, stride=2,padding=1)
        self.res_out_spatial_conv2d = conv_s_p(self.batchNorm, in_planes=512, out_planes=256, kernel_size=4, stride=2,padding=1)

        self.pretrained = pretrained
        self.init_std = init_std


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m, std=self.init_std)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')    
    ############

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)


    def forward(self, input, output_layers=None, image_resize=288, sp_threshold=0.75):
        # pdb.set_trace()
        B,C,N,H,W=input.shape
 
        res_img = F.interpolate(input, size = [self.clip_len, 240, 240], mode='trilinear') #torch.Size([50, 3,8, 240, 240])
        res_temporal_img = res_img.permute(0,2,1,3,4).reshape(self.clip_len*B,3,240, 240)   #rch.Size([400, 3, 240, 240])

        spatial_temporal_input = self.resnet18_feature_extractor(res_temporal_img)#torch.Size([400, 512, 8, 8])
    
        # ##################################################################
        # ###########temporal
            
        temporal_input = self.res_out_temporal_conv2d(spatial_temporal_input).reshape(B,self.clip_len,16,4,4).reshape(B,self.clip_len,-1) #torch.Size([50, 8, 2048])
        temporal_input_1 = self.pos_drop(temporal_input + self.pos_embed)
        temporal_form = temporal_input_1 + self.drop_path1(self.ls1(self.attn1(self.norm1(temporal_input_1)))) 
        temporal_form = self.norm2(temporal_form) 
        temporal_form_out = temporal_form + self.drop_path2(self.ls2(self.mlp1(temporal_form)))#torch.Size([50, 8, 256])
      

        # #################################################################
        ###########spatial
        
        spatial_input = self.res_out_spatial_conv2d(spatial_temporal_input).reshape(B,self.clip_len,256,4,4).reshape(B,self.clip_len,-1) #torch.Size([50, 8, 2048])

        spatial_input= spatial_input.sum(1)
        spatial_input = spatial_input.reshape(B,256,-1).permute(0,2,1)  

        spatial_input_1 = self.pos_drop(spatial_input + self.pos_embed_spatial)

        spatial_form = spatial_input_1 + self.drop_path1(self.ls3(self.attn2(self.norm3(spatial_input_1)))) 
        spatial_norm = self.norm4(spatial_form) 
        spatial_form_out = spatial_form + self.drop_path2(self.ls4(self.mlp2(spatial_form)))
        
        ########################################################################
        ####fusion
        temporal_spatial_agg = torch.cat((temporal_form_out,spatial_form_out),1) 
        # temporal_spatial_agg = self.norm3(temporal_spatial_agg)
        temporal_spatial_agg1 = temporal_spatial_agg + self.drop_path1(self.ls5(self.attn3(self.norm5(temporal_spatial_agg)))) 
        temporal_spatial_agg1 = self.norm6(temporal_spatial_agg1)
        temporal_spatial_agg1_out =temporal_spatial_agg1 + self.drop_path2(self.ls6(self.mlp3(temporal_spatial_agg1))) 

        temporal_agg = temporal_spatial_agg1_out[:,:self.clip_len,:]  
        spatial_agg = temporal_spatial_agg1_out[:,self.clip_len:,:] 

        ###########################################################################
        #####
        temporal_feature = temporal_form_out + temporal_agg
        spatial_feature  = spatial_form_out + spatial_agg

    
        temporal_feature_former = temporal_feature + self.drop_path1(self.ls7(self.attn4(self.norm7(temporal_feature)))) 
        temporal_feature_former = self.norm8(temporal_feature_former)
        temporal_feature_former_out = temporal_feature_former + self.drop_path2(self.ls8(self.mlp4(temporal_feature_former)))

        spatial_feature_former = spatial_feature + self.drop_path1(self.ls9(self.attn5(self.norm9(spatial_feature))))
        spatial_feature_former = self.norm10(spatial_feature_former) 
        spatial_feature_former_out = spatial_feature_former + self.drop_path2(self.ls10(self.mlp5(spatial_feature_former)))
   
        ##############################################################################
        #####cat
        cat_spatial_temporal = torch.cat((temporal_feature_former_out,spatial_feature_former_out),1)

        fusionFeats = torch.flatten(cat_spatial_temporal, start_dim=1, end_dim=2) 
        
        predict = self.snn_fc(fusionFeats) 

        return predict 
