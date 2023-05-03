"""
Code for running the DNN Only model of the Domain Adaptation QSAR model
SR GAN will inherit from this class
"""

import random
from collections import defaultdict
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch import nn


### move these over to the SR-GAN.DomAdpQSAR folder to setup SRGAN model
from DomAdpQSAR.models import Classifier
from DomAdpQSAR.data_test_utils import QSARDataset

from DomAdpQSAR.srgan import feature_corrcoef

# import DomApdQSAR functions here

from DomAdpQSAR.utility import gpu, seed_all


from DomAdpQSAR.dnn import DnnExperiment
# from srgan import Experiment

# from QSARsettings import Settings



class DomAdpQSARDNN(DnnExperiment):
    """A Class to manage an experimental trial with only a DNN."""
    def __init__(self, settings):
        super().__init__(settings)
        self.labeled_criterion = nn.BCELoss() # CE for classifcation
        self.layer_sizes = self.settings.layer_sizes
        self.federated_dataframe = pd.read_pickle(self.settings.federated_datapath)
        self.clean_dataframe = pd.read_pickle(self.settings.clean_datapath)
        self.validation_dataframe = pd.read_pickle(self.settings.validation_datapath)
        self.test_dataframe = pd.read_pickle(self.settings.test_datapath)
        # define datasets and data loaders here
        self.federated_dataset = None
        self.clean_dataset = None
        # self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        self.federated_dataset_loader = None
        self.clean_dataset_loader = None
        #to set the current training loader being useds
        self.train_dataset_loader = None
        self.validation_dataset_loader = None
        self.test_dataset_loader = None

        self.examples = None

        self.rank = self.settings.rank

    def model_setup(self):
        """Sets up the model."""
        self.DNN = Classifier(self.layer_sizes)

    def dataset_setup(self):
        """Sets up the datasets for the application."""

        rank = self.rank
        settings = self.settings

        self.federated_dataset = QSARDataset(self.federated_dataframe, dataset_size=settings.federated_dataset_size, rank=rank)
        self.clean_dataset = QSARDataset(self.clean_dataframe, dataset_size=settings.labeled_dataset_size, rank=rank)
        self.validation_dataset = QSARDataset(self.validation_dataframe, dataset_size=settings.validation_dataset_size)
        self.test_dataset = QSARDataset(self.test_dataframe, dataset_size=settings.test_dataset_size)

        self.federated_dataset_loader = DataLoader(self.federated_dataset, batch_size=settings.federated_batch_size, shuffle=True)
        self.clean_dataset_loader = DataLoader(self.clean_dataset, batch_size=settings.batch_size, shuffle=True)
        self.validation_dataset_loader = DataLoader(self.validation_dataset, batch_size=settings.batch_size, shuffle=True)
        self.test_dataset_loader = DataLoader(self.test_dataset, batch_size=settings.batch_size, shuffle=True)


    def dnn_loss_calculation(self, labeled_examples, labels):
        """Calculates the labeled loss."""
        predicted_logits = self.DNN(labeled_examples)

        if self.rank is not None:
            y, r = labels
            labeled_loss = self.labeled_criterion(predicted_logits, y)
            labeled_loss = torch.mean(labeled_loss * r)
        else:
            labeled_loss = self.labeled_criterion(predicted_logits, labels)

        labeled_loss *= self.settings.labeled_loss_multiplier
        return labeled_loss
    
    def evaluate(self, data_loader, network, summary_writer, summary_name, step):
        """Evaluates the model on the test dataset."""
        # self.model_setup()
        # self.load_models()
        self.eval_mode()

        #create list of labels from test dataset indexes
        labels = []
        predictions = []
        for data in data_loader:
            try:
                x, y = data
            except:
                x, y, _ = data

            prediction = network(x)
            # print("prediction mean: ", prediction.mean())
            # extend instead of append because we need to flatten the list
            predictions.extend(prediction)
            labels.extend(y)

        # convert to tensors
        predictions = torch.tensor(predictions)
        labels = torch.tensor(labels)


        # compute the accuracy 
        acc_count = self.compute_ACC(predictions, labels)
        # compute the BAC
        bac_count = self.compute_BAC(predictions, labels)
        
        print("Evaluating on {} dataset:".format(summary_name))
        print("Accuracy: ", acc_count)
        print("BAC: ", bac_count)

        #write to summary writer
        if summary_writer is not None:
            summary_writer.add_scalar(summary_name + "{}/Accuracy".format(summary_name), acc_count, step)
            summary_writer.add_scalar(summary_name + "{}/BAC".format(summary_name), bac_count, step)

        return acc_count, bac_count
    
    def validation_summaries(self, step: int):
        """Prepares the summaries that should be run for the given application."""
        summary_writer = self.dnn_summary_writer

        self.evaluate(self.train_dataset_loader, self.DNN, summary_writer, "Training", step)
        self.evaluate(self.validation_dataset_loader, self.DNN, summary_writer, "Validation", step)
        self.evaluate(self.test_dataset_loader, self.DNN, summary_writer, "Test", step)

    


    def predict_activiity(self, fp):
        """Predicts the activity of a given fingerprint."""

        self.inference_setup()

        fp = torch.from_numpy(fp).float().to(gpu)
        activity = self.DNN(fp)
        return activity
        
    def compute_BAC(self, predictions, labels):
        """Computes the Balanced Accuracy for the given predictions and labels."""
        # Convert predictions to binary values (0 or 1)
        binary_predictions = torch.round(predictions)

        # Compute true positives, true negatives, false positives, and false negatives
        true_positives = torch.sum(torch.logical_and(binary_predictions == 1, labels == 1))
        true_negatives = torch.sum(torch.logical_and(binary_predictions == 0, labels == 0))
        false_positives = torch.sum(torch.logical_and(binary_predictions == 1, labels == 0))
        false_negatives = torch.sum(torch.logical_and(binary_predictions == 0, labels == 1))

        # print("True Positives: ", true_positives)
        # print("True Negatives: ", true_negatives)
        # print("False Positives: ", false_positives)
        # print("False Negatives: ", false_negatives)


        # Compute balanced accuracy using torch tensor division
        sensitivity = true_positives.float() / (true_positives + false_negatives).float()
        specificity = true_negatives.float() / (true_negatives + false_positives).float()
        balanced_accuracy = (sensitivity + specificity) / 2

        return balanced_accuracy

    def compute_ACC(self, predictions, labels):
        """Computes the Accuracy (ACC) for the given predictions and labels."""
        # Convert predictions to binary values (0 or 1)
        binary_predictions = torch.round(predictions)
        # print("Binary Predictions: ", binary_predictions)
        # print("Labels: ", labels)
        # Compute the number of correct predictions
        correct_predictions = torch.sum(binary_predictions == labels)

        # Compute the total number of predictions
        total_predictions = labels.size(0)
        # print("Total Predictions: ", total_predictions)
        # print("Correct Predictions: ", correct_predictions)
        # Compute the accuracy
        accuracy = correct_predictions / total_predictions

        return accuracy

        
    def dnn_training_step(self, examples, labels, step):
        """Runs an individual round of DNN training."""
        # self.DNN.apply(disable_batch_norm_updates)  # No batch norm
        # if self.examples is None:
        #     print("Examples is None")
        #     self.examples = examples

        # for idx, selfex in enumerate(self.examples):
        #     ex = examples[idx]
        #     print(ex, selfex)
        #     if ex != selfex:
        #         print("Examples not equal")
        #         print("ex: ", ex)
        #         print("selfex: ", selfex)
        print("Step: ", step, "No. Ex: ", examples.size(), end='\r')
        self.train_mode()
        # check if example = example
        self.dnn_summary_writer.step = step
        self.dnn_optimizer.zero_grad()
        # print("Examples size: ", examples.size())
        dnn_loss = self.dnn_loss_calculation(examples, labels)
        self.examples = examples

        # print("DNN loss: ", dnn_loss)
        dnn_loss.backward()
        self.dnn_optimizer.step()
        # Summaries.
        if self.dnn_summary_writer.is_summary_step():
            self.dnn_summary_writer.add_scalar('Discriminator/Labeled Loss', dnn_loss.item())
            if hasattr(self.DNN, 'features') and self.DNN.features is not None:
                self.dnn_summary_writer.add_scalar('Feature Norm/Labeled', self.DNN.features.norm(dim=1).mean().item(),step)
                self.dnn_summary_writer.add_image('Feature Norm/Labeled', plot_to_image(self.DNN.features))
    



    def optimizer_to_gpu(self, optimizer):
        """Moves the optimizer to GPU."""
        """changed to skip as no cuda on mac"""

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    try:
                        state[k] = v.cuda()
                    except:
                        pass


    def set_rank(self, rank):
        """Sets the rank of the current experiment."""
        self.rank = rank
        self.dataset_setup()
        if self.rank is not None:
            self.labeled_criterion = nn.BCELoss(reduction='none')
        elif self.rank is None:
            self.labeled_criterion = nn.BCELoss()







import io
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.image as mpimg

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PyTorch tensor
    and returns it. The supplied figure is closed and inaccessible after this call.
    """
    # Convert the figure to a NumPy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image_np = np.array(Image.open(buf))

    # Use torchvision.transforms to convert the NumPy array to a PyTorch tensor
    image_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = image_transform(image_np)

    # Add the batch dimension using unsqueeze
    # image = image.unsqueeze(0)

    return image

def log_feature_plot(features):
    corr = feature_corrcoef(features.detach())
    imgplot = plt.imshow(corr)
    image = plot_to_image(imgplot)
    return image
    