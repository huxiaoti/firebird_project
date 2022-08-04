# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size) # (224, 224)
        patch_size = to_2tuple(patch_size) # (4, 4)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        # 取整除 - 返回商的整数部分（向下取整）9//2 = 4
        # [56, 56]
        self.img_size = img_size # (224, 224)
        self.patch_size = patch_size # (4, 4)
        self.patches_resolution = patches_resolution # [56, 56]
        self.num_patches = patches_resolution[0] * patches_resolution[1] # 56 * 56 = 3136

        self.in_chans = in_chans # 3
        self.embed_dim = embed_dim # 96

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.proj = nn.Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x) # (batch, 96, 224/4=56, 224/4=56)
        x = x.flatten(2) # 把 H, W 高和宽平铺开 batch, 96, 56, 56 --> batch, 96, 56*56=3136
        x = x.transpose(1, 2)  # 把 channel 通道 特征放到最后 batch, 3136, 96
        if self.norm is not None:
            x = self.norm(x)
        return x