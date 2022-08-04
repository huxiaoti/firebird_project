# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:
    :math:`\\sigma(x) = \\frac{1}{1 + \\exp(-k(x-x_0))}`
    :param x0: The value of the sigmoid midpoint
    :type x0: float
    :param k: The slope of the sigmoid - trainable -  :math:`k \\geq 0`
    :type k: float
    :param train: Whether :math:`k` is a trainable parameter
    :type train: bool
    """

    def __init__(self, num_genes):
        super().__init__()

        self.num_genes = num_genes
        
        self.k = nn.Parameter(torch.empty(num_genes, 1)) # 负 k 为负相关
        self.a = nn.Parameter(torch.empty(num_genes, 1))
        self.b = nn.Parameter(torch.empty(num_genes, 1)) # b 不要，从 y=1 开始

        # self.to_kab = nn.Linear(dim_input, 3 * dim_output, bias=False)
        # k, a, b = self.to_kab(x).chunk(3, dim=-1)

        self.reset_parameters()

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.k.data, gain=1)
        nn.init.xavier_uniform_(self.a.data, gain=1)
        nn.init.xavier_uniform_(self.b.data, gain=1)


    def forward(self, x):
        """
        Applies the function to the input elementwise

        x.shape: [batch, 1] --> [batch, num_genes, 1]
        output.shape: [batch, num_genes, 1]
        """

        # k = torch.clamp(self.k, min=1e-5)
        a = torch.clamp(self.a, min=1e-5)

        x = x.unsqueeze(1).repeat(1, self.num_genes, 1)
        
        output = self.k / (1 + torch.exp(-a * x))

        return output