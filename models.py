
import os
from abc import ABC, abstractmethod

import torch
import torch.optim as optim
import copy
import numpy as np
from torch.nn.modules.module import Module
from torch import nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader

import glob

import datetime
import math

from data_test_utils import MyDataset, dataset_compiler, plot_losses, compute_BAC, compute_ACC

### Thsese models come from CHatGPT so need to be corrected to work

### Model layers need names for the model to be saved and loaded correctly - esp if we are using multiple models

# original MLP from the paper ### TODO change features to be optional as an input



class Classifier(torch.nn.Module):
    def __init__(self, layersize=[2**11, 2**11, 2**9, 2**7, 2**0], dropout=0.33):
        super(Classifier, self).__init__()
        self.hidden = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        self.dropout = dropout

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
        for idx, layer in enumerate(self.hidden):
            # print(f"hidden layer {idx} output shape: {x.shape}")
            x = F.relu(self.hidden[idx](x))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.batchnorm[idx](x)

            if idx == len(self.hidden) - 1:
                last_hidden = x  # save activation of last hidden layer
        # print(f"output layer input shape: {x.shape}")
        # print(f"output layer output shape: {self.output(x).shape}")
        output = torch.sigmoid(self.output(x))  # apply sigmoid activation to output layer for binary classification
    
        return output






class Generator(torch.nn.Module):
    def __init__(self,  layersize=[100, 128, 256, 1024, 1024]):
        super(Generator, self).__init__()
        self.hidden = nn.ModuleList()
        
        for k in range(len(layersize)):
            self.hidden.append(nn.Linear(layersize[k], layersize[k+1]))

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        return x







class MLP(torch.nn.Module):
    def __init__(self, layersize=[1024, 1024, 256, 128], dropout=0.33):
        super(MLP, self).__init__()
        self.hidden = nn.ModuleList()
        self.batchnorm = nn.ModuleList()
        self.dropout = dropout
        for k in range(len(layersize) - 1):
            self.hidden.append(nn.Linear(layersize[k], layersize[k + 1]))
            self.batchnorm.append(nn.BatchNorm1d(layersize[k + 1]))
        self.output = nn.Linear(layersize[-1], 1)  # output layer for regression

    def forward(self, x):
        for layer in range(len(self.hidden)):
            x = F.relu(self.hidden[layer](x))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.batchnorm[layer](x)
            if layer == len(self.hidden) - 2:
                last_hidden = x  # save activation of last hidden layer
        output = self.output(x)  # apply linear transformation to output layer
        return output, last_hidden



# class Generator(torch.nn.Module):
#     def __init__(self, input_size=100, output_size=1024, layersize=[128, 512, 1024]):
#         super(Generator, self).__init__()
#         self.hidden = nn.ModuleList()
        
#         self.hidden.append(nn.Linear(input_size, layersize[0]))

#         for k in range(1, len(layersize)-1):
#             self.hidden.append(nn.Linear(layersize[k], layersize[k+1]))
#         self.output = nn.Linear(layersize[-1], output_size)

#     def forward(self, x):
#         for layer in self.hidden:
#             x = F.relu(layer(x))
#         x = self.output(x)
#         return x






def feature_extractor(model, x):
    ### TODO change this to select which layer to extract features from
    # Set model to evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        _, features = model(x)

    return features




def train_model(model, train_loader, val_loader, num_epochs=1, lr=0.001, weight_decay=0.0, rank=None, device=None):
    # Define loss function and optimizer
    criterion = nn.BCELoss() # CE for classifcation
    # criterion = nn.MSELoss() # MSE for regression
    if device is None:
        device = 'cpu'
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Track losses and accuracies
    train_losses = []
    val_losses = []
    val_accuracies = []

    # clear axes for plotting later
    ax = None
    print(device)


    model.to(device)

    # Train loop
    for epoch in range(num_epochs):
        # Set model to train mode
        model.train()

        # Train on batches
        train_loss = 0
        if rank is not None:
            criterion = nn.BCELoss(reduction='none')
            for data in train_loader:
                x, y, r = data
                # x, y, r = x.to(device), y.to(device), r.to(device)
                
                
                # Forward pass
                outputs = model(x)
                # reshape y to match output shape
                y = y.reshape(outputs.shape)
                # loss 
                loss = criterion(outputs, y)

                loss = torch.mean(loss * r)
                # print(loss)
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss = np.mean(train_loss)
            # train_loss /= len(train_loader)
            train_losses.append(train_loss)


        elif rank is None:
            criterion = nn.BCELoss() # CE for classifcation

            for data in train_loader:
                # print(str(epoch)+":"+str(idx), end=)
                x, y = data
                # Move data to GPU
                # x, y = x.to(device), y.to(device)


                # Forward pass
                outputs = model(x)
                # reshape y to match output shape
                y = y.reshape(outputs.shape)
                # loss 
                loss = criterion(outputs, y)
                # print(loss)
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Compute average training loss for epoch
            train_loss = np.mean(train_loss)
            # train_loss /= len(train_loader)
            train_losses.append(train_loss)




        # Set model to evaluation mode
        model.eval()
        criterion = nn.BCELoss()

        # Evaluate on validation set
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            BAC_count = []
            for data in val_loader:
                # Forward pass
                # print(x.shape)
                x, y = data

                outputs = model(x)
                
                predicted = outputs.squeeze()  # Round the outputs to obtain binary predictions
                # print(predicted.round(), y)
                balanced_accuracy = compute_BAC(predicted, y)

                y = y.reshape(outputs.shape)

                # Compute loss
                loss = criterion(outputs, y)

                val_loss += loss.item()


                correct += (predicted == y).sum().item()
                total += y.size(0)

                BAC_count.append(balanced_accuracy)
            BAC_count = np.mean(BAC_count)
            
            val_accuracies.append(BAC_count)
            accuracy = correct / total
            # print(correct, total)
            # print(f"Val accuracy: {accuracy:.4f}")


        # Compute average validation loss for epoch

        # val_loss = np.mean(val_loss)
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        val_accuracy = np.mean(val_accuracies)

        # Print loss for epoch
        print(f"Epoch {epoch + 1}: Train loss = {train_loss:.4f}, Val loss = {val_loss:.4f}, Val BAC = {balanced_accuracy:.4f}")


        # plot_losses([train_losses, val_losses])

    # Print final losses
    print(f"Final: Train loss = {train_loss:.4f}, Val loss = {val_loss:.4f}, Val BAC = {val_accuracy:.4f}")
    # save model

    # move losses to cpu
  

    return model, train_losses, val_losses



def compile_and_train_dataset(model, federated_data, clean_data, validation_loader, percentages, step, rank, gradual_FT_params):

    compiled_dataset = dataset_compiler(F_dataset=federated_data, S0_dataset=clean_data, percentages=percentages, rank=rank)
    compiled_loader = MyDataset(compiled_dataset, rank=rank)
    compiled_loader = DataLoader(compiled_loader, batch_size=gradual_FT_params["batch_size"], shuffle=True)

    temp_params = copy.deepcopy(gradual_FT_params)
    temp_params["experiment_name"] = gradual_FT_params["experiment_name"] + "_step_{}".format(step)
    
    Grad_FT_model, train_losses, val_losses = train_model(model, compiled_loader, validation_loader,
                                                        num_epochs=gradual_FT_params["max_epochs"],
                                                        lr=gradual_FT_params["learning_rate"], 
                                                        weight_decay=gradual_FT_params["weight_decay"], 
                                                        rank=rank)

    plot_losses([train_losses, val_losses], title="Losses on compiled dataset" + " " + temp_params["experiment_name"])

    save_model(model, params=temp_params)

    return Grad_FT_model, train_losses, val_losses


def gradual_fine_tuning(model, federated_data, clean_data, validation_loader, base=2, base_params=None, gradual_FT_params=None, rank=None):
    # Gradual fine-tuning testing - each fine-tuning step is 1/2 the size of the previous step for the federated dataset
    # Each step we compile a dataset with the federated data and the clean data and then train the model on that dataset

    Grad_FT_model = None


    # Make directory for checkpoints
    os.makedirs(gradual_FT_params["checkpoint_dir"], exist_ok=True)

    number_of_gradual_steps = int(math.log(math.floor(len(federated_data) / len(clean_data)), base))



    for i in range(number_of_gradual_steps):
        if Grad_FT_model is not None:
            model = Grad_FT_model
        i = i + 1
        federated_percentage = 1 / (base ** i)
        clean_percentage = 1
        percentages = [federated_percentage, clean_percentage]
        
        
        Grad_FT_model, train_losses, val_losses = compile_and_train_dataset(model, federated_data, clean_data, validation_loader, percentages,i,rank,gradual_FT_params)


    return Grad_FT_model, train_losses, val_losses









# save model
import datetime
import glob
def save_model(model, params=None, path=None):
    """Save model dict to file, use parameter dictionary to save to path, 
    if not save to current directory of the current date and time"""
    
    current_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = "model"
    if path is not None:
        prefix = path
    
    if path is None:
        prefix = os.getcwd()

    if params is not None:
        name = params["experiment_name"]
        prefix = params["checkpoint_dir"]


    path = os.path.join(prefix, name+"_"+current_date_time)

    path = path + '.pt'
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    return path



def load_model(model, path, latest=False):
    """Load model state dict from file, 
    if latest is True, load latest model from directory"""
    if latest:
        path = max(glob.glob(path + "/*.pt"), key=os.path.getctime)
        print(f"Loading latest model from {path}")

    model.load_state_dict(torch.load(path))

    return model