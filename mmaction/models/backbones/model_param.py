
import torchvision

# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import math
from torch.nn.init import kaiming_normal_, constant_

import matplotlib.pyplot as plt 

import torch.nn.functional as F
import numpy as np 
import cv2 


from collections import OrderedDict

import warnings 
warnings.filterwarnings("ignore")
import pdb 
import matplotlib.pyplot as plt

# from ...utils import get_root_logger
# from mmcv.utils import _BatchNorm
# from mmcv.cnn import ConvModule, constant_init, kaiming_init, normal_init

# from .resnet import ResNet

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from mmaction.models.backbones import torchvision_resnet 
import random
from torchstat import stat
import torchvision.models as models
from ptflops import get_model_complexity_info
from thop import profile

####################################
def conv_s_p(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        )

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
        dim=256
        num_heads = 4
        self.num_heads = 4
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


class Time_Space_Agg(nn.Module):
    expansion = 1
    # default_hyper_params = dict(pretrain_model_path="", crop_pad=4, pruned=True,)
    def __init__(self, pretrained=None, batchNorm=True, output_layers=None, init_std=0.05,
                dim=24, clip_len=9, num_heads=6, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None, drop_path=0.,to_device="cuda:2"):  
        super(Time_Space_Agg, self).__init__()
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
        dim=256
        time_dim=dim
        space_dim = dim


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
        self.norm1 = nn.LayerNorm(time_dim)
        self.norm2 = nn.LayerNorm(space_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm5 = nn.LayerNorm(dim)


        self.attn1 = Attention(time_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,to_device=self.device)
        self.attn2 = Attention(space_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,to_device=self.device)
        self.attn3 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,to_device=self.device)
        self.attn4 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,to_device=self.device)
        self.attn5 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,to_device=self.device)

        self.ls1 =  nn.Identity()
        self.ls2 = nn.Identity()
        self.ls3 = nn.Identity()
        self.ls4 = nn.Identity()
        self.ls5 = nn.Identity()
        self.ls6 = nn.Identity()
        self.ls7 = nn.Identity()
        self.ls8 = nn.Identity()
        self.ls9 = nn.Identity()
        self.ls10 =nn.Identity()



        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp_voxel = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        self.mlp1 = Mlp(in_features=time_dim, hidden_features=int(time_dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        self.mlp2 = Mlp(in_features=space_dim, hidden_features=int(space_dim * mlp_ratio), act_layer=nn.GELU, drop=drop)
        self.mlp3 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        self.mlp4 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        self.mlp5 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        embed_dim=256
        self.num_tokens = 1
        drop_rate = 0.
        self.clip_len = 8
        
        # self.patch_embed1 = PatchEmbed(
        #     img_size=240, patch_size=30, in_chans=3, embed_dim=128)

        # self.patch_embed2 = PatchEmbed(
        #     img_size=240, patch_size=60, in_chans=3, embed_dim=256)
        
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        # self.pos_embed = nn.Parameter(torch.zeros(1, 8, 1024))

        self.pos_embed = nn.Parameter(torch.zeros(1, self.clip_len, time_dim))

        self.pos_embed_space = nn.Parameter(torch.zeros(1, 16, space_dim))
 
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.resnet18_feature_extractor = torchvision_resnet.resnet18(pretrained = True)

        self.res_out_time_conv2d = conv_s_p(self.batchNorm, in_planes=512, out_planes=16, kernel_size=4, stride=2,padding=1)
        self.res_out_space_conv2d = conv_s_p(self.batchNorm, in_planes=512, out_planes=256, kernel_size=4, stride=2,padding=1)

        self.pretrained = pretrained
        self.init_std = init_std


    # def init_weights(self):
    #     """Initiate the parameters either from existing checkpoint or from
    #     scratch."""
        
    #     if isinstance(self.pretrained, str):
    #         logger = get_root_logger()
    #         logger.info(f'load model from: {self.pretrained}')

    #         load_checkpoint(self, self.pretrained, strict=False, logger=logger)

    #     elif self.pretrained is None:
    #         for m in self.modules():
    #             if isinstance(m, nn.Conv3d):
    #                 kaiming_init(m)
    #             elif isinstance(m, nn.Linear):
    #                 normal_init(m, std=self.init_std)
    #             elif isinstance(m, _BatchNorm):
    #                 constant_init(m, 1)

    #     else:
    #         raise TypeError('pretrained must be a str or None')    
    # ############

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)


    def forward(self, input, output_layers=None, image_resize=288, sp_threshold=0.75):
        B,C,N,H,W=input.shape
 
        res_img = F.interpolate(input, size = [self.clip_len, 240, 240], mode='trilinear') #torch.Size([50, 3,8, 240, 240])
        res_time_img = res_img.permute(0,2,1,3,4).reshape(self.clip_len*B,3,240, 240)   #rch.Size([400, 3, 240, 240])
        # res_space_img = res_img[:,:,self.clip_len-1,:,:]    #torch.Size([50, 3, 240, 240])

        # space_time_input = self.resnet18_feature_extractor(res_time_img)#torch.Size([400, 512, 8, 8])
        space_time_input = self.resnet18_feature_extractor(res_time_img)#torch.Size([400, 512, 8, 8])
    
        # ##################################################################
        # ###########time
            
        time_input = self.res_out_time_conv2d(space_time_input).reshape(B,self.clip_len,16,4,4).reshape(B,self.clip_len,-1) #torch.Size([50, 8, 2048])
        time_input_1 = self.pos_drop(time_input + self.pos_embed)
        time_norm = self.norm1(time_input_1) 
        time_form = time_norm + self.drop_path1(self.ls1(self.attn1(time_norm))) 
        time_form_out = time_form + self.drop_path2(self.ls2(self.mlp1(time_form)))#torch.Size([50, 8, 256])
      

        # #################################################################
        ###########space
     
        # space_input = self.patch_embed1(res_time_img).reshape(B,self.clip_len,64,128).sum(1)
        
        space_input = self.res_out_space_conv2d(space_time_input).reshape(B,self.clip_len,256,4,4).reshape(B,self.clip_len,-1) #torch.Size([50, 8, 2048])

        space_input= space_input.sum(1)
        space_input = space_input.reshape(B,256,-1).permute(0,2,1)    #torch.Size([50, 64, 32]) 

        space_input_1 = self.pos_drop(space_input + self.pos_embed_space)

        space_norm = self.norm2(space_input_1) 
        space_form = space_norm + self.drop_path1(self.ls3(self.attn2(space_norm))) 
        space_form_out = space_form + self.drop_path2(self.ls4(self.mlp2(space_form)))#torch.Size([50, 16, 256])
        # space_form_out = self.space_agg_fc(space_form_out)#torch.Size([50, 64, 128])
        
        ########################################################################
        ####cross
        # space_form_out1 = self.norm3(space_form_out)#50,8,512
        # time_form_out1 = self.norm4(time_form_out1)#50,8,512


        # x_space_time_agg1 = space_form_out1 + self.drop_path1(self.ls5(self.attn3(space_form_out1,time_form_out1))) #torch.Size([30, 8, 1536])
        # x_space_time_agg11 =x_space_time_agg1 + self.drop_path2(self.ls6(self.mlp3(x_space_time_agg1))) #torch.Size([30, 8, 1536])

        # x_space_time_agg2 = time_form_out1 + self.drop_path1(self.ls7(self.attn4(time_form_out1,space_form_out1))) #torch.Size([30, 8, 1536])
        # x_space_time_agg22 = x_space_time_agg2 + self.drop_path2(self.ls8(self.mlp4(x_space_time_agg2)))

        # x_stage_agg_out = torch.cat((x_space_time_agg11,x_space_time_agg22),1)
        
        time_space_agg = torch.cat((time_form_out,space_form_out),1) 
        time_space_agg = self.norm3(time_space_agg)
        time_space_agg1 = time_space_agg + self.drop_path1(self.ls5(self.attn3(time_space_agg))) 
        time_space_agg1_out =time_space_agg1 + self.drop_path2(self.ls6(self.mlp3(time_space_agg1))) 

        time_agg = time_space_agg1_out[:,:self.clip_len,:]  #torch.Size([10, 8, 256])
        space_agg = time_space_agg1_out[:,self.clip_len:,:] #torch.Size([10, 16, 256])

        ###########################################################################
        #####
        time_feature = time_form_out + time_agg
        space_feature  = space_form_out + space_agg

        time_feature_norm = self.norm4(time_feature)
        time_feature_former = time_feature_norm + self.drop_path1(self.ls7(self.attn4(time_feature_norm))) 
        time_feature_former_out = time_feature_former + self.drop_path2(self.ls8(self.mlp4(time_feature_former)))

        space_feature_norm = self.norm5(space_feature)
        space_feature_former = space_feature_norm + self.drop_path1(self.ls9(self.attn5(space_feature_norm))) 
        space_feature_former_out = space_feature_former + self.drop_path2(self.ls10(self.mlp5(space_feature_former)))


        # time_feature_former_out = time_feature_former_out+time_feature
        # space_feature_former_out = space_feature_former_out+ space_agg       
        ##############################################################################
        #####cat
        cat_space_time = torch.cat((time_feature_former_out,space_feature_former_out),1)
        # cat_space_time = torch.cat((time_feature,space_feature),1)

        
        eventFeats = torch.flatten(cat_space_time, start_dim=1, end_dim=2) #torch.Size([20, 9248])
        
        predict = self.snn_fc(eventFeats)   ## torch.Size([8, 800, 72, 72]) torch.Size([20, 4096])

        return predict 


def get_parameter_number(model):
    pdb.set_trace()
    
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ =='__main__':
    # model = Time_Space_Agg()
    # # model = models.alexnet()
    # # stat(model, (3, 224, 224))
    # total_num,trainable_num= get_parameter_number(model)   

    # # 打印模型参数
    # #for param in model.parameters():
    #     #print(param)
    
    # #打印模型名称与shape
    # i=0
    # for name,parameters in model.named_parameters():
    #     print(name,':',parameters.size())
    #     i=i+1
    # print('number of parameters: ',i)
    # print("--------------------end-------------------------")



    # ####################################################
    # ####mac numberparams
    # with torch.cuda.device(0):
    #       net = Time_Space_Agg()
    #       pdb.set_trace()

          # macs, params = get_model_complexity_info(net, (3,8, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
          # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
          # print('{:<30}  {:<8}'.format('Number of parameters: ', params))



    #######################################################
    ####flops
    pdb.set_trace()
    model = Time_Space_Agg()
    input = torch.randn(2, 3, 8, 224, 224)
    flops, params = profile(model, inputs=(input, ))
    # print('flops:', flops)
    # print('params:', params)

    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')