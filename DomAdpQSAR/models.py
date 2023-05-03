"""Model architectures code."""

import torch
import torch.nn as nn
from torch.nn import Module, Linear
import torch.nn.functional as F

from DomAdpQSAR.utility import gpu, seed_all


class Classifier(Module):
    """MLP Based Classifier"""
    def __init__(self, layersize=[2**11, 2**11, 2**9, 2**7, 2**0], dropout=0.33):
        super().__init__()
        seed_all(0)
        self.hidden = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        self.dropout = dropout
        self.features = None

        for idx, layer in enumerate(layersize[:-2]):
            self.hidden.append(nn.Linear(layersize[idx], layersize[idx+1]))
            self.batchnorm.append(nn.BatchNorm1d(layersize[idx+1]))
        self.output = nn.Linear(layersize[-2], layersize[-1])  # output layer for binary classification


    def forward(self, x):
        for idx, layer in enumerate(self.hidden):
            x = F.relu(self.hidden[idx](x))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.batchnorm[idx](x)
            self.features = x
          
        output = torch.sigmoid(self.output(x))  # apply sigmoid activation to output layer for binary classification
    
        return output.squeeze()


class Generator(Module):
    """MLP Based Generator"""
    def __init__(self, layersize=[2**6, 2**7, 2**9, 2**11, 2**11], dropout=0.33):
        super().__init__()
        seed_all(0)
        self.hidden = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        self.dropout = dropout
        self.features = None

        for idx, layer in enumerate(layersize[:-2]):
            self.hidden.append(nn.Linear(layersize[idx], layersize[idx+1]))
            self.batchnorm.append(nn.BatchNorm1d(layersize[idx+1]))

        self.output = nn.Linear(layersize[-2], layersize[-1])  # output layer for binary classification


    def forward(self, x):
        for idx, layer in enumerate(self.hidden):
            # print(f"hidden layer {idx} output shape: {x.shape}")
            x = F.relu(self.hidden[idx](x))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.batchnorm[idx](x)
            self.features = x
          
        output = torch.sigmoid(self.output(x))  # apply sigmoid activation to output layer for binary classification
    
        return output


