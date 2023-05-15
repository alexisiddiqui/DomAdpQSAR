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
            # unindent this line to stop it from adding for every layer
        self.features = x
          
        output = torch.sigmoid(self.output(x))  # apply sigmoid activation to output layer for binary classification
    
        return output.squeeze()





class TF_Classifier(torch.nn.Module):
    def __init__(self, layersize=[2**5, 2**3, 2**10, 2**6, 2**0], dropout=0.33, featuriser:Classifier=None):
        super(TF_Classifier, self).__init__()
        self.hidden = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        self.dropout = dropout
        self.featuriser = featuriser
        self.features = None

        for idx, layer in enumerate(layersize[:-2]):
            self.hidden.append(nn.Linear(layersize[idx], layersize[idx+1]))
            self.batchnorm.append(nn.BatchNorm1d(layersize[idx+1]))

        self.output = nn.Linear(layersize[-2], layersize[-1])  # output layer for binary classification


        # save names for each layer
        for idx, layer in enumerate(self.hidden):
            self.hidden[idx].name = f"hidden_{idx}"
        for idx, layer in enumerate(self.batchnorm):
            self.batchnorm[idx].name = f"batchnorm_{idx}"
        self.output.name = "output"



    def forward(self, x):
        # if self.featuriser is not None:
        _ = self.featuriser(x)
        x = self.featuriser.features

        for idx, layer in enumerate(self.hidden):
            # print(f"hidden layer {idx} output shape: {x.shape}")
            x = F.relu(self.hidden[idx](x))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.batchnorm[idx](x)
        self.features = x

        # print(f"output layer input shape: {x.shape}")
        # print(f"output layer output shape: {self.output(x).shape}")
        output = torch.sigmoid(self.output(x))  # apply sigmoid activation to output layer for binary classification
    
        return output.squeeze()

class Generator(Module):
    """MLP Based Generator"""
    def __init__(self, layersize=[2**5, 2**7, 2**9, 2**11, 2**11], dropout=0.33):
        super().__init__()
        seed_all(0)
        self.hidden = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        self.dropout = dropout
        self.features = None
        self.input_size = layersize[0]
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
        # round to 0 or 1
        output = torch.round(output)
        return output



# function to freeze the layers of a model up to a certain layer index
def freeze_layers(model, layer_index):
    """
    Takes in a model and a layer index, and freezes the hidden layers up to that index
    """
    # accounts for layers and activations are in the same list as well as for 0 indexing
    layer_index = ((layer_index +1)*2)
    for i, param in enumerate(model.hidden.parameters()):
        # i = i+1
        if i < layer_index:
            param.requires_grad = False

    for i, param in enumerate(model.batchnorm.parameters()):
        # i = i+1
        if i < layer_index:
            param.requires_grad = False

    
    return model            
