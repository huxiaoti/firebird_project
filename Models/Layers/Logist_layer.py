# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:
    """

    def __init__(self, num_genes):
        super().__init__()

        self.num_genes = num_genes
        
        self.k = nn.Parameter(torch.ones(1))
        self.a = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.ones(1))
        

    def forward(self, x):
        """
        Applies the function to the input elementwise

        x.shape: [batch, 1] --> [batch, num_genes, 1]
        output.shape: [batch, num_genes, 1]
        """

        # a = torch.clamp(self.a, min=1e-5)

        # x = x.unsqueeze(1).repeat(1, self.num_genes, 1)
        
        output = (self.k + 1) / (torch.exp(-self.a * x) + self.k*self.b)

        return output
