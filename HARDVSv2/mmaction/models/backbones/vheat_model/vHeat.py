import time
import math
from functools import partial
from typing import Optional, Callable

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1).contiguous()
    
    
def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')
    
    
class StemLayer(nn.Module):
    r""" Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      'channels_first')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PolicyNet(nn.Module):
    def __init__(self, in_dim):
        super(PolicyNet, self).__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.BatchNorm1d(32),
            nn.Linear(32, 3))

    def forward(self, x, temp):
        logits = self.net(x)
        hard_mask = F.gumbel_softmax(logits, tau=temp, hard=True, dim=-1)
        return hard_mask

class DynamicConv(nn.Module):
    def __init__(self, in_channels):
        super(DynamicConv, self).__init__()
        self.conv =  nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channels),
            )

    def forward(self, x):
        return self.conv(x)
        
        
class Heat2D(nn.Module):
    """
    du/dt -k(d2u/dx2 + d2u/dy2) = 0;
    du/dx_{x=0, x=a} = 0
    du/dy_{y=0, y=b} = 0
    =>
    A_{n, m} = C(a, b, n==0, m==0) * sum_{0}^{a}{ sum_{0}^{b}{\phi(x, y)cos(n\pi/ax)cos(m\pi/by)dxdy }}
    core = cos(n\pi/ax)cos(m\pi/by)exp(-[(n\pi/a)^2 + (m\pi/b)^2]kt)
    u_{x, y, t} = sum_{0}^{\infinite}{ sum_{0}^{\infinite}{ core } }
    
    assume a = N, b = M; x in [0, N], y in [0, M]; n in [0, N], m in [0, M]; with some slight change
    => 
    (\phi(x, y) = linear(dwconv(input(x, y))))
    A(n, m) = DCT2D(\phi(x, y))
    u(x, y, t) = IDCT2D(A(n, m) * exp(-[(n\pi/a)^2 + (m\pi/b)^2])**kt)
    """    
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        
        self.policy = PolicyNet(hidden_dim)
        self.dynamicconv = DynamicConv(hidden_dim*2)
        
    def infer_init_heat2d(self, freq):
        weight_exp = self.get_decay_map((self.res, self.res), device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False)
        del self.to_k

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        # cos((x + 0.5) / N * n * \pi) which is also the form of DCT and IDCT
        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * f(x) )
        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * F(n) )
        # returns: (Res_n, Res_x)
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        # exp(-[(n\pi/a)^2 + (m\pi/b)^2])
        # returns: (Res_h, Res_w)
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight
    
    def forward_hco(self, x, freq_embed=None):
        B, H, W, C = x.shape

        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
            assert weight_cosn is not None
            assert weight_cosm is not None
            assert weight_exp is not None
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]
                
        x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
        x = F.conv1d(x.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)
        
        if self.infer_mode:
            x = torch.einsum("bnmc,nmc->bnmc", x, self.k_exp)
        else:
            weight_exp = torch.pow(weight_exp[:, :, None], self.to_k(freq_embed))
            x = torch.einsum("bnmc,nmc -> bnmc", x, weight_exp) # exp decay
        
        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1))
        x = F.conv1d(x.contiguous().view(-1, M, C), weight_cosm.t().contiguous().view(W, M, 1)).contiguous().view(B, H, W, -1)
        
        return x
        
    def forward(self, x: torch.Tensor, B, freq_embed_rgb=None, freq_embed_event=None):
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous()) # B, H, W, 2C
        x, z = x.chunk(chunks=2, dim=-1) # B, H, W, C
                
        x_split  = torch.chunk(x.contiguous().view((B, 16,) + x.shape[1:]), 2, dim=1)
        rgb_input, event_input = x_split[0], x_split[1]
        x_rgb = rgb_input.contiguous().view((-1, ) + rgb_input.shape[2:])
        x_event = event_input.contiguous().view((-1, ) + event_input.shape[2:])
        
        ###############
        z_split  = torch.chunk(z.contiguous().view((B, 16,) + z.shape[1:]), 2, dim=1)
        rgb_z, event_z = z_split[0], z_split[1]
        z_rgb = rgb_z.contiguous().view((-1, ) + rgb_z.shape[2:])
        z_event = event_z.contiguous().view((-1, ) + event_z.shape[2:])
        ###############
        
        out_rgb = self.forward_hco(x_rgb, freq_embed_rgb)
        out_event = self.forward_hco(x_event, freq_embed_event) 
        
        ### 
        rgb1 = out_rgb.contiguous().view((B, 8,) + out_rgb.shape[1:])
        event1 = out_event.contiguous().view((B, 8,) + out_event.shape[1:])
        x1 = torch.cat((rgb1, event1), dim=1)
        x1 = x1.contiguous().view((-1, ) + x1.shape[2:])
        policy_out = self.policy(x1.mean(dim=(1,2)), 1)
        
        ## 
        x1 = x1
        
        ## 
        intersection = out_rgb * out_event 
        rgb2 = out_rgb - intersection
        event2 = out_event - intersection
        rgb2 = rgb2.contiguous().view((B, 8,) + rgb2.shape[1:])
        event2 = event2.contiguous().view((B, 8,) + event2.shape[1:])
        x2 = torch.cat((rgb2, event2), dim=1) 
        x2 = x2.contiguous().view((-1, ) + x2.shape[2:]) 

        ## 
        CAT_rgbe = torch.cat((out_rgb, out_event), dim=-1)
        CAT_rgbe = CAT_rgbe.permute(0, 3, 1, 2).contiguous()
        rgbe_w = torch.sigmoid(self.dynamicconv(CAT_rgbe)).permute(0, 2, 3, 1).contiguous()
        rgb_w, event_w = torch.chunk(rgbe_w, 2, dim=-1)
        rgb3 = out_rgb * rgb_w
        event3 = out_event * event_w
        rgb3 = rgb3.contiguous().view((B, 8,) + rgb3.shape[1:])
        event3 = event3.contiguous().view((B, 8,) + event3.shape[1:])
        x3 = torch.cat((rgb3, event3), dim=1) 
        x3 = x3.contiguous().view((-1, ) + x3.shape[2:]) 

        x = torch.cat((x1.unsqueeze(dim=1), x2.unsqueeze(dim=1), x3.unsqueeze(dim=1)), dim=1)
        x = torch.einsum("brnmc,br -> bnmc", x, policy_out)
        
        #############
        x_split  = torch.chunk(x.contiguous().view((B, 16,) + x.shape[1:]), 2, dim=1)
        rgb_input, event_input = x_split[0], x_split[1]
        x_rgb = rgb_input.contiguous().view((-1, ) + rgb_input.shape[2:])
        x_event = event_input.contiguous().view((-1, ) + event_input.shape[2:])
        
        x_rgb = self.out_norm(x_rgb)
        x_event = self.out_norm(x_event)
        
        x_rgb = x_rgb * nn.functional.silu(z_rgb)
        x_event = x_event * nn.functional.silu(z_event)
        
        x_rgb = self.out_linear(x_rgb)
        x_event = self.out_linear(x_event)
        
        x_rgb = x_rgb.permute(0, 3, 1, 2).contiguous()
        x_event = x_event.permute(0, 3, 1, 2).contiguous()
        
        x_rgb = x_rgb.contiguous().view((B, 8,) + x_rgb.shape[1:])
        x_event = x_event.contiguous().view((B, 8,) + x_event.shape[1:])
        x_rgbe = torch.cat((x_rgb, x_event), dim=1)
        x = x_rgbe.contiguous().view((-1, ) + x_rgbe.shape[2:])
        #############
        
        # x = self.out_norm(x)
        # x = x * nn.functional.silu(z)
        
        # x = self.out_linear(x)

        # x = x.permute(0, 3, 1, 2).contiguous()

        return x


class HeatBlock(nn.Module):
    def __init__(
        self,
        res: int = 14,
        infer_mode = False,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        mlp_ratio: float = 4.0,
        post_norm = True,
        layer_scale = None,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.op = Heat2D(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=True)
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None
        
        self.infer_mode = infer_mode
        
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)

    # def _forward(self, x: torch.Tensor, freq_embed):
    def _forward(self, x: torch.Tensor, B, freq_embed_rgb, freq_embed_event):
        if self.post_norm:          
            x = x + self.drop_path(self.gamma1[:, None, None] * self.norm1(self.op(x, B, freq_embed_rgb, freq_embed_event))) 
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.norm2(self.mlp(x))) # FFN
        
        return x

    
    def forward(self, input: torch.Tensor, B, freq_embed_rgb=None, freq_embed_event=None): # torch.Size([b*8, 96, 56, 56])

        return self._forward(input, B, freq_embed_rgb, freq_embed_event)


class AdditionalInputSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self[:-1]:
            if isinstance(module, nn.Module):
                x = module(x, *args, **kwargs)
            else:
                x = module(x)
        x = self[-1](x)
        return x


class vHeat(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], drop_path_rate=0.2, patch_norm=True, post_norm=True,
                 layer_scale=None, use_checkpoint=False, mlp_ratio=4.0, img_size=224,
                 act_layer='GELU', infer_mode=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        
        self.depths = depths
        
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=self.embed_dim,
                                     act_layer='GELU',
                                     norm_layer='LN')
        
        res0 = img_size/patch_size
        self.res = [int(res0), int(res0//2), int(res0//4), int(res0//8)]
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        self.infer_mode = infer_mode
        
        self.freq_embed = nn.Parameter(torch.zeros(self.res[0], self.res[0], self.dims[0]), requires_grad=True)
        trunc_normal_(self.freq_embed, std=.02)
        self.freq_embed2 = nn.Parameter(torch.zeros(self.res[0], self.res[0], self.dims[0]), requires_grad=True)
        trunc_normal_(self.freq_embed2, std=.02)
        
        
        self.freq_conv1_rgb = nn.Sequential(
            nn.Conv2d(self.dims[0], self.dims[1], kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(self.dims[1]),
        )
        self.freq_conv1_event = nn.Sequential(
            nn.Conv2d(self.dims[0], self.dims[1], kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(self.dims[1]),
        )
        
        self.freq_conv2_rgb = nn.Sequential(
            nn.Conv2d(self.dims[1], self.dims[2], kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(self.dims[2]),
        )
        self.freq_conv2_event = nn.Sequential(
            nn.Conv2d(self.dims[1], self.dims[2], kernel_size=3, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(self.dims[2]),
        )
        
        self.freq_conv3_rgb = nn.Sequential(
            nn.Conv2d(self.dims[2], self.dims[3], kernel_size=3, stride=2, padding=1, bias=False),  
            nn.BatchNorm2d(self.dims[3]),
        )    
        self.freq_conv3_event = nn.Sequential(
            nn.Conv2d(self.dims[2], self.dims[3], kernel_size=3, stride=2, padding=1, bias=False),  
            nn.BatchNorm2d(self.dims[3]),
        )    
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(self.make_layer(
                res = self.res[i_layer],
                dim = self.dims[i_layer],
                depth = depths[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=LayerNorm2d,
                post_norm=post_norm,
                layer_scale=layer_scale,
                downsample=self.make_downsample(
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=LayerNorm2d,
                ) if (i_layer < self.num_layers - 1) else nn.Identity(),
                mlp_ratio=mlp_ratio,
                infer_mode=infer_mode,
            ))   
            
        self.classifier = nn.Sequential(
            LayerNorm2d(self.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            # nn.Linear(self.num_features, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def make_downsample(dim=96, out_dim=192, norm_layer=LayerNorm2d):
        return nn.Sequential(
            #norm_layer(dim),
            #nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_dim)
        )

    @staticmethod
    def make_layer(
        res=14,
        dim=96, 
        depth=2,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=LayerNorm2d,
        post_norm=True,
        layer_scale=None,
        downsample=nn.Identity(), 
        mlp_ratio=4.0,
        infer_mode=False,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(HeatBlock(
                res=res,
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                post_norm=post_norm,
                layer_scale=layer_scale,
                infer_mode=infer_mode,
            ))
        
        return AdditionalInputSequential(
            *blocks, 
            downsample,
        )
 
    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless
        
        Conv2D is not intialized !!!
        """
        # print(m, getattr(getattr(m, "weight", nn.Identity()), "INIT", None), isinstance(m, nn.Linear), "======================")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def infer_init(self):
        for i, layer in enumerate(self.layers):
            for block in layer[:-1]:
                block.op.infer_init_heat2d(self.freq_embed[i])
                
        del self.freq_embed
    
    def forward_features(self, x, B):
        x = self.patch_embed(x)  # torch.Size([b*8, 96, 56, 56])
        
        for i, layer in enumerate(self.layers):
            x_split  = torch.chunk(x.contiguous().view((B, 16,) + x.shape[1:]), 2, dim=1)
            x_rgb, x_event = x_split[0], x_split[1]
            x_rgb = x_rgb.contiguous().view((-1, ) + x_rgb.shape[2:])
            x_event = x_event.contiguous().view((-1, ) + x_event.shape[2:]) 
            freq_embed_rgb = x_rgb.permute(0, 2, 3, 1).contiguous().mean(dim=0).cuda()
            trunc_normal_(freq_embed_rgb, std=.02)
            freq_embed_event = x_event.permute(0, 2, 3, 1).contiguous().mean(dim=0).cuda()
            trunc_normal_(freq_embed_event, std=.02)
             
            if i == 0:
                self.freq_embed_rgb = nn.Parameter(self.freq_embed * freq_embed_rgb)
                self.freq_embed_event = nn.Parameter(self.freq_embed2 * freq_embed_event)
            else:
                if i == 1:
                    self.freq_embed_rgb = nn.Parameter(self.freq_conv1_rgb(self.freq_embed_rgb.unsqueeze(0).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().squeeze(0))
                    self.freq_embed_event = nn.Parameter(self.freq_conv1_event(self.freq_embed_event.unsqueeze(0).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().squeeze(0))
                if i == 2:
                    self.freq_embed_rgb = nn.Parameter(self.freq_conv2_rgb(self.freq_embed_rgb.unsqueeze(0).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().squeeze(0))
                    self.freq_embed_event = nn.Parameter(self.freq_conv2_event(self.freq_embed_event.unsqueeze(0).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().squeeze(0))
                if i == 3:
                    self.freq_embed_rgb = nn.Parameter(self.freq_conv3_rgb(self.freq_embed_rgb.unsqueeze(0).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().squeeze(0))
                    self.freq_embed_event = nn.Parameter(self.freq_conv3_event(self.freq_embed_event.unsqueeze(0).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().squeeze(0))
                
                self.freq_embed_rgb = nn.Parameter(self.freq_embed_rgb * freq_embed_rgb)
                self.freq_embed_event = nn.Parameter(self.freq_embed_event * freq_embed_event)
            
            x = layer(x, B, self.freq_embed_rgb, self.freq_embed_event) 
            # (B, C, H, W)  torch.Size([b*8, 96, 56, 56]) → torch.Size([b*8, 192, 28, 28]) → torch.Size([b*8, 384, 14, 14]) → torch.Size([b*8, 768, 7, 7])
               
        return x

    def forward(self, x):  # torch.Size([b, 8, 3, 224, 224])
        inputs_rgb = x[0]
        inputs_event = x[1]
        B = inputs_rgb.shape[0]
        
        x = torch.cat((inputs_rgb, inputs_event), dim=1)
        x = x.contiguous().view((-1, ) + x.shape[2:])
        
        x = self.forward_features(x, B)  
        
        x = self.classifier(x)
        
        return x


if __name__ == "__main__":
    from fvcore.nn import flop_count_table, flop_count_str, FlopCountAnalysis
    model = vHeat().cuda()
    input = torch.randn((1, 3, 224, 224), device=torch.device('cuda'))
    analyze = FlopCountAnalysis(model, (input,))
    print(flop_count_str(analyze))



