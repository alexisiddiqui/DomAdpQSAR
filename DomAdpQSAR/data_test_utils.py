### Data and testing functions

# import FLuID as fluid
from typing import Any
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import calculate_tanimoto_similarity, calculate_target_similarity

from DomAdpQSAR.utility import gpu
 
### TODO - update to use params

# 


### Original plan was to draw our own domains from the data we have but we are just going to use the splits from fluid notebook 

## Need to find a way to save the data out from the fluid notebook and load it in a seperate notebook for model running

# We do need to generate tran/val splits for our generated datasets

from torch.utils.data import Dataset



class QSARDataset(Dataset):
    def __init__(self, dataframe, rank=None, dataset_size=None, device=gpu):
        if dataset_size is not None or 0:
            self.dataframe = dataframe
        else:
            self.dataframe = dataframe.sample(n=dataset_size, random_state=42)
        self.fp = self.dataframe['FP'].to_numpy()
        self.labels = self.dataframe['CLASS'].to_numpy()

        fp = self.fp
        fp = [f.astype(np.float32)for f in fp]
        fp = np.array(fp)
        fp = torch.from_numpy(fp)

        self.FP = fp
        labels = self.labels
        labels = [l.astype(np.float32) for l in labels]
        labels = np.array(labels)
        labels = torch.from_numpy(labels)

        self.LABELS = labels



        # self.FP = torch.from_numpy(self.fp).to(device)
        # self.LABELS = torch.from_numpy(self.labels).to(device)
        self.ranks = None
        if rank is not None:
            self.ranks = self.dataframe[rank].to_numpy()
        if device is None:
            self.device = 'cpu'
        else:
            self.device = device

    def __len__(self):
        return len(self.dataframe)
    
    def __call__(self):
        return self.dataframe

    def __getitem__(self, index):

        x = torch.tensor(self.fp[index], dtype=torch.float32, device=self.device)
        y = torch.tensor(self.labels[index], dtype=torch.float32, device=self.device)
        
        if self.ranks is not None:
            r = torch.tensor(self.ranks[index], dtype=torch.float32, device=self.device)
            return x, y, r
        else: 
            return x, y


def split_data(dataset, params=None):
    """
    Splits the dataset into train and validation sets
    Ingores any idea of overlap between the domains but does ensure that the validation set is not in the training set
    Inputs:
    - dataset ... dataset to be split
    - params ... dictionary of parameters
    Outputs:
    - train ... training set
    - val ... validation set
    """

    try:
        frac=params['validation_ratio'] 
        random_state=params['random_state']
    except KeyError:
        print("frac and random_state not specified in params - using defaults")
        frac=0.2 
        random_state=42
    except:
        raise ValueError("validation_ratio and random_state must be specified in params")

    validationData = dataset.sample(frac=frac, random_state=random_state)


    trainingData = dataset[~dataset.INCHI.isin(validationData.INCHI)].dropna()

    return trainingData, validationData


from torch.utils.data import Dataset

#define a function that compiles the datasets based on specifications that we set for various combinations of domains (F, S0, T)
def dataset_compiler(F_dataset=None, S0_dataset=None, target_dataset=None, percentages=None, rank=None, random_state=42):
    """
    Compiles the datasets into a single dataset that can then be loaded into the model

    Inputs: 
    - F_dataset ... Federated dataset
    - S0_dataset ... Source dataset 
    - target_dataset ... Target dataset

    Parameters:
    - percentages ... list of percentages for the federated and source datasets
    - rank ... Choose rankings for the datasets when sampling

    Outputs:
    - dataset ... compiled dataset as a pandas dataframe
    """
    
    # check that the datasets are present
    datasets = []
    for dataset in [F_dataset, S0_dataset, target_dataset]:
        if dataset is not None:
            datasets.append(dataset)

    if all(x is None for x in [F_dataset, S0_dataset, target_dataset]):
        raise ValueError("No datasets have been specified")
    
    # check that the percentages are present if not set to 100%
    if percentages is None:
        percentages = [1]*len(datasets)

    if len(datasets) != len(percentages):
        raise ValueError("The number of datasets must match the number of percentages")
        

    # rank based on the specified ranking
    if rank is not None:
        #check if rank is in the dataset
        for dataset in datasets:
            if rank not in dataset.columns and target_dataset is not None:
                print("Rank {} not in dataset, calculating ranks".format(rank))
                try:
                    # calculate the ranks
                    dataset[rank] = calculate_target_similarity(dataset, target_dataset, simi_type=rank, mean="mean")
                except:
                    raise ValueError("The rank is not in the dataset and cannot be calculated")

            dataset.sort_values(by=rank, ascending=False, inplace=True)
    

    sampled_datasets = []
    # sample the datasets based on the percentages
    for dataset, percentage in zip(datasets, percentages):
        print("Initial size of dataset: {}".format(len(dataset)))
        print("Sampling {}% of the dataset".format(percentage*100))
        # if ranked sample the top fraction
        if rank is not None:
            sampled_datasets.append(dataset.head(int(len(dataset)*percentage)))
        else:
            # if not ranked sample randomly
            sampled_datasets.append(dataset.sample(frac=int(percentage*100)/100, random_state=random_state))
        print("Final size of dataset: {}".format(len(dataset)))



    # combine the datasets
    compiled_dataset = pd.concat(sampled_datasets, axis=0, ignore_index=True)
    # print(compiled_dataset.head())
    print("Compiled dataset size: {}".format(len(compiled_dataset)))
    return compiled_dataset




### Redo this to cover the full range of statistics - ac
def get_domain_adaptation_stats(model, F_test, S0_test, target_test, SMILES=False):

    """
    Calculates the percentage error for the domain adaptation model
    Inputs:
    - model ... regression model ### how to set this to evaluate mode
    - F_test ... Federated test(val) dataset
    - S0_test ... Source test(val) dataset
    - target_test ... Target test dataset - final test set
    -> all datasets contain a SMILES string/FP and a target value
    Outputs:
    - stats ... dictionary of error metrics
    """
    
    if SMILES is True:
        raise NotImplementedError("SMILES is not implemented yet, please run fingerprints first and set SMILES to False")
    
    stats = {}

    # loop through all test sets # TODO make this work with actual model
    for test in [F_test, S0_test, target_test]:
        # loop through each compound in the current test set
        for FP in test:
            # calculate the predicted value for the current compound
            pred = model.predict(FP)
            # calculate the error between the predicted and actual value
            error = abs(pred - test[FP])
            # calculate the percentage error
            error = error / test[FP]

            # add the error to the list of errors for the current test set
            stats[test].append(error)

    # calculate the mean error for each test set
    stats['F_test_mean'] = np.mean(stats['F_test'])
    stats['S0_test_mean'] = np.mean(stats['S0_test'])
    stats['target_test_mean'] = np.mean(stats['target_test'])
    # print means on one line
    print("F_test_mean: {}, S0_test_mean: {}, target_test_mean: {}".format(stats['F_test_mean'], stats['S0_test_mean'], stats['target_test_mean']))
    return stats


def plot_losses(losses, logscale=True, ax=None, title=None, save=False, filename=None):
    """
    Plots the losses for the model
    Inputs:
    - losses ... dictionary of losses
    - ax ... existing matplotlib axes object to plot on (default=None)
    - title ... title of the plot (default=None)
    - save ... boolean to save the plot (default=False)
    - filename ... name of the file to save the plot as (default=None)
    Outputs:
    - plot of the losses
    """
    if title is None:
        title = "Losses on current dataset"
    
    if ax is None:
        ax = plt.gca()


    # check for convergence
    conv = 0.99
    convergence = []
    for idx,_ in enumerate(losses[0]):
        convergence.append(conv**(idx))
    pre = ''
    if logscale is True:
        losses = np.log(losses)
        convergence = np.log(convergence)
        pre = 'log '
    # plot the losses
    ax.plot(losses[0], label='train')
    ax.plot(losses[1], label='val')
    ax.plot(convergence, label='exp decay '+str(conv)+'^epoch')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(pre+'Loss')
    # ax.set_ylim([-8, 3])
    ax.legend()
    if save is True and filename is not None:
        plt.savefig(filename+'.png')

    plt.show()

    return ax
# visualise domain adaptation stats

def plot_domain_adaptation_stats(stats, title=None, save=False, filename=None):
    """
    Plots the mean domain adaptation stats for the model
    Inputs:
    - stats ... dictionary of stats
    - title ... title of the plot
    - save ... boolean to save the plot
    - filename ... name of the file to save the plot as
    Outputs:
    - plot of the stats
    """
    if title is None:
        title = "Mean Domain Adaptation Stats"
    # plot the stats
    plt.plot(stats['F_test_mean'], label='F_test')
    plt.plot(stats['S0_test_mean'], label='S0_test')
    plt.plot(stats['target_test_mean'], label='target_test')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    if save is True and filename is not None:
        # save the plot
        plt.savefig(filename+'.png')
        # save the stats
        stats = pd.DataFrame(stats)
        # only keep the means
        stats = stats[stats.index.str.contains('mean')]
        stats.to_csv(filename+'.csv')

    plt.show()


