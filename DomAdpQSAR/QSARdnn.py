"""
Code for running the DNN Only model of the Domain Adaptation QSAR model
SR GAN will inherit from this class
"""

import random
from collections import defaultdict
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import math
import datetime

### move these over to the SR-GAN.DomAdpQSAR folder to setup SRGAN model
from DomAdpQSAR.models import Classifier, freeze_layers
from DomAdpQSAR.data_test_utils import QSARDataset, dataset_compiler

from DomAdpQSAR.srgan import feature_corrcoef

# import DomApdQSAR functions here

from DomAdpQSAR.utility import gpu, seed_all, make_directory_name_unique


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
        self.compiled_dataset = None

        self.federated_dataset_loader = None
        self.clean_dataset_loader = None
        #to set the current training loader being useds
        self.train_dataset_loader = None
        self.validation_dataset_loader = None
        self.test_dataset_loader = None
        self.compiled_dataset_loader = None

        self.examples = None

        self.rank = self.settings.rank
        ### can probs keep this in the settings rather than including it here
        # self.gradual_base = self.settings.gradual_base
        # self.number_of_gradual_steps = self.settings.number_of_gradual_steps
        

    def model_setup(self):
        """Sets up the model."""
        
        self.DNN = Classifier(self.layer_sizes)
        self.freeze_DNN_layers()

    def dataset_setup(self):
        """Sets up the datasets for the application."""

        rank = self.rank
        print("dataset rank: ", rank)
        settings = self.settings

        self.federated_dataset = QSARDataset(self.federated_dataframe, 
                                             dataset_size=settings.federated_dataset_size, 
                                             rank=rank)
        self.clean_dataset = QSARDataset(self.clean_dataframe, 
                                         dataset_size=settings.labeled_dataset_size, 
                                         rank=rank)
        self.validation_dataset = QSARDataset(self.validation_dataframe, 
                                              dataset_size=settings.validation_dataset_size)
        self.test_dataset = QSARDataset(self.test_dataframe, 
                                        dataset_size=settings.test_dataset_size)

        self.federated_dataset_loader = DataLoader(self.federated_dataset, 
                                                   batch_size=settings.federated_batch_size, 
                                                   shuffle=True)
        self.clean_dataset_loader = DataLoader(self.clean_dataset, 
                                               batch_size=settings.batch_size, 
                                               shuffle=True)
        self.validation_dataset_loader = DataLoader(self.validation_dataset, 
                                                    batch_size=settings.batch_size, 
                                                    shuffle=True)
        self.test_dataset_loader = DataLoader(self.test_dataset, 
                                              batch_size=settings.batch_size, 
                                              shuffle=True)



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
        print("Step: ", step, "No. Ex: ", examples.size(), end='\r')
        self.train_mode()

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
                self.dnn_summary_writer.add_scalar('Feature Norm/Labeled', 
                                                   self.DNN.features.norm(dim=1).mean().item(),step)
                self.dnn_summary_writer.add_image('Feature Corr/Labeled', 
                                                  summwriter_feature_plot(self.DNN.features), step)
    



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

    def set_summary_step(self, data_loader=None):
        """Sets the summary step to the size of the training data."""
        if data_loader is None:
            data_loader = self.train_dataset_loader
        self.settings.summary_step_period = len(data_loader)
        print(self.settings.summary_step_period)

    def set_steps_to_run(self, steps_to_run=None):
        """Sets the number of steps to run."""
        if steps_to_run is not None:
            self.settings.steps_to_run = steps_to_run
        else :
            self.settings.steps_to_run = self.settings.epochs_to_run * len(self.train_dataset_loader)
        
    
    def set_training_data(self, data_loader):
        """Sets the training data loader."""
        self.train_dataset_loader = data_loader
        self.set_summary_step(data_loader=self.train_dataset_loader)
        self.set_steps_to_run()


    def gradual_fine_tune(self, rank=None):
        """Gradually fine-tunes the DNN on the compiled labelled data. Defaults to no subsampling."""

        self.trial_directory = os.path.join(self.settings.logs_directory, self.settings.trial_name)
        if (self.settings.skip_completed_experiment and os.path.exists(self.trial_directory) and
                '/check' not in self.trial_directory and not self.settings.continue_existing_experiments):
            print('`{}` experiment already exists. Skipping...'.format(self.trial_directory))
            return
        if not self.settings.continue_existing_experiments:
            self.trial_directory = make_directory_name_unique(self.trial_directory)
        else:
            if os.path.exists(self.trial_directory) and self.settings.load_model_path is not None:
                raise ValueError('Cannot load from path and continue existing at the same time.')
            elif self.settings.load_model_path is None:
                self.settings.load_model_path = self.trial_directory
            elif not os.path.exists(self.trial_directory):
                self.settings.continue_existing_experiments = False
        print(self.trial_directory)
        os.makedirs(os.path.join(self.trial_directory, self.settings.temporary_directory), exist_ok=True)
        self.prepare_summary_writers()

        self.set_rank(rank)

        seed_all(0)

        # self.dataset_setup()
        self.model_setup()
        self.prepare_optimizers()
        self.load_models()
        self.gpu_mode()
        self.train_mode()

        if self.settings.number_of_gradual_steps is None:
            number_of_gradual_steps = int(math.log(math.floor(
                len(self.federated_dataframe) / len(self.clean_dataframe)), 
                self.settings.gradual_base))
        else:
            number_of_gradual_steps = self.settings.number_of_gradual_steps
        print("Number of gradual steps: ", number_of_gradual_steps)

        step = 0 + self.starting_step
        for i in range(number_of_gradual_steps):
            i = i + 1
            federated_percentage = 1 / (self.settings.gradual_base ** i)
            clean_percentage = 1
            percentages = [federated_percentage, clean_percentage]
            # print("Percentages: ", percentages)
            compiled_dataframe = dataset_compiler(F_dataset=self.federated_dataframe, 
                                                     S0_dataset=self.clean_dataframe, 
                                                     percentages=percentages,
                                                     rank=self.rank)
            # print(compiled_dataframe.columns)
            # if self.settings.use_rank_in_GFT_step is False and self.rank is not None:
            #     DS_include_rank = None
            #     print("DS include rank: ", DS_include_rank)
            # else:
            #     DS_include_rank = self.rank
            print(self.rank)
            self.compiled_dataset = QSARDataset(compiled_dataframe,
                                                dataset_size=0,
                                                rank=self.rank)
            
            print("Compiled dataset: ", len(self.compiled_dataset))

            self.compiled_dataset_loader = DataLoader(self.compiled_dataset, 
                                                      batch_size=self.settings.batch_size, 
                                                      shuffle=True)
            # print("Compiled dataset loader: ", len(self.compiled_dataset_loader))

            self.set_training_data(self.compiled_dataset_loader)

            step_time_start = datetime.datetime.now()

            for epoch in range(self.settings.gradual_epochs):
                # hacking the epoch to be the step
                epoch = epoch * self.settings.summary_step_period
                

                for samples in self.compiled_dataset_loader:
                    if self.settings.use_rank_in_GFT_step is False and self.rank is not None:
                        x, y, _ = samples
                        examples, labels = x, (y, torch.ones_like(y).to(gpu))
                    else:
                        try:
                            x, y, z = samples
                            examples, labels = x, (y, z)
                        except:
                            examples, labels = samples
                    

                    # print(samples)
                    step += 1
                    # This is to account for subsampling and not using rank in the loss calculation
                    
                    self.dnn_training_step(examples, labels, step)
                    if self.dnn_summary_writer.is_summary_step() or step == self.settings.steps_to_run - 1:
                        print('\rStep {}, {}...'.format(step, datetime.datetime.now() - step_time_start), end='')
                        step_time_start = datetime.datetime.now()
                        self.eval_mode()
                        with torch.no_grad():
                            self.validation_summaries(step)
                        self.train_mode()
                    self.handle_user_input(step)

        # return back to the original ranking from settings
        self.set_rank(self.settings.rank)
        print('Completed {}'.format(self.trial_directory))
        if self.settings.should_save_models:
            self.save_models(step=step)

        

    def freeze_DNN_layers(self, layer_index=None):
        """Freezes all layers up to the specified index."""
        if layer_index is None:
            layer_index = self.settings.freeze_layers
        freeze_layers(self.DNN, layer_index)





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
    fig = plt.gcf()
    plt.close(fig)
    # plt.close(figure)
    buf.seek(0)
    image_np = np.array(Image.open(buf))

    # Use torchvision.transforms to convert the NumPy array to a PyTorch tensor
    image_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_np = image_transform(image_np)

    # Add the batch dimension using unsqueeze
    # image = image.unsqueeze(0)

    return image_np

def summwriter_feature_plot(features):
    """Plots the features as a correlation matrix
    and returns the image for summary writer"""
    corr = feature_corrcoef(features.detach().cpu())
    imgplot = plt.imshow(corr)
    image = plot_to_image(imgplot)
    return image
    