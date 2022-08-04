# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

def MSELoss(predict, label):

    loss = nn.MSELoss()(predict, label)

    return loss

# l1_loss
# 可以加一个基于基因上下调的二分类 focial loss