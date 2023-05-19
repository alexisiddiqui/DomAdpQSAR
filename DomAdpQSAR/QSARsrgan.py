
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
from scipy.stats import norm, wasserstein_distance
from recordclass import RecordClass

import copy
import torch
from torch import nn
import math
import datetime

### move these over to the SR-GAN.DomAdpQSAR folder to setup SRGAN model
from DomAdpQSAR.models import Classifier, Generator, TF_Classifier, freeze_layers
from DomAdpQSAR.data_test_utils import QSARDataset, dataset_compiler
from DomAdpQSAR.presentation import generate_display_frame
from DomAdpQSAR.QSARdnn import DomAdpQSARDNN
from DomAdpQSAR.srgan import feature_covariance_loss, feature_corrcoef, feature_angle_loss, disable_batch_norm_updates

# import DomApdQSAR functions here

from DomAdpQSAR.utility import gpu, seed_all, make_directory_name_unique, MixtureModel, SummaryWriter, standard_image_format_to_tensorboard_image_format
from DomAdpQSAR.QSARdnn import summwriter_feature_plot

from DomAdpQSAR.dnn import DnnExperiment
from DomAdpQSAR.srgan import Experiment


class DomAdpQSARSRGAN(Experiment):
    """A Class to manage an experimental trial with the SR-GAN model."""
    def __init__(self, settings):
        super().__init__(settings)
        self.labeled_criterion = nn.BCELoss() # CE for classifcation
        # self.layer_sizes = self.settings.layer_sizes
        self.federated_dataframe = pd.read_pickle(self.settings.federated_datapath)
        self.clean_dataframe = pd.read_pickle(self.settings.clean_datapath)
        self.validation_dataframe = pd.read_pickle(self.settings.validation_datapath)
        self.test_dataframe = pd.read_pickle(self.settings.test_datapath)
        # define datasets and data loaders here
        ### Commented as federated - unlabelled, clean - labelled
        # self.federated_dataset = None
        # self.clean_dataset = None
        self.train_dataset = None
        # self.validation_dataset = None
        # self.test_dataset = None
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
        self.featuriser = None

    def load_featuriser(self, path=None):
        settings = copy.deepcopy(self.settings)
        settings.layer_sizes[2] = 2**6
        featuriser = DomAdpQSARDNN(settings)
        if path is None:
            featuriser.settings.load_model_path = self.settings.load_featuriser_path
        else:
            featuriser.settings.load_model_path = path
        print("Loading featuriser...")
        print(f"Featuriser path: {featuriser.settings.load_model_path}")

        self.model_setup(featuriser=featuriser)

        print("Featuriser loaded.")

    def model_setup(self, featuriser: DomAdpQSARDNN=None):
        if featuriser is not None:
            print("Setting up SR-GAN with featuriser...1")
            featuriser.model_setup()
            self.featuriser = featuriser.eval_mode()
            self.featuriser = featuriser.DNN

        if self.featuriser is not None:
            print("Setting up SR-GAN with featuriser...2")
            self.DNN = TF_Classifier(self.settings.transfer_layer_sizes, featuriser=self.featuriser)
            self.D = TF_Classifier(self.settings.transfer_layer_sizes, featuriser=self.featuriser)
        else:  
            self.DNN = Classifier(self.settings.layer_sizes)
            self.D = Classifier(self.settings.layer_sizes)

        self.G = Generator(self.settings.generator_layer_sizes)


    def dataset_setup(self):
        """Sets up the datasets for the application."""

        rank = self.rank
        print("dataset rank: ", rank)
        settings = self.settings

        self.unlabeled_dataset = QSARDataset(self.federated_dataframe, 
                                             dataset_size=settings.federated_dataset_size, 
                                             rank=rank,
                                             device=gpu)
        self.train_dataset = QSARDataset(self.clean_dataframe, 
                                         dataset_size=settings.labeled_dataset_size, 
                                         rank=rank,
                                            device=gpu)
        
        self.validation_dataset = QSARDataset(self.validation_dataframe, 
                                              dataset_size=settings.validation_dataset_size, 
                                              device=gpu)
        
        self.test_dataset = QSARDataset(self.test_dataframe, 
                                        dataset_size=settings.test_dataset_size,
                                        device=gpu)

        self.unlabeled_dataset_loader = DataLoader(self.unlabeled_dataset, 
                                                   batch_size=settings.batch_size, 
                                                   shuffle=True)
        self.train_dataset_loader = DataLoader(self.train_dataset, 
                                               batch_size=settings.batch_size, 
                                               shuffle=True)
        self.validation_dataset_loader = DataLoader(self.validation_dataset, 
                                                    batch_size=settings.batch_size, 
                                                    shuffle=True)
        self.test_dataset_loader = DataLoader(self.test_dataset, 
                                              batch_size=settings.batch_size, 
                                              shuffle=True)
        self.set_summary_step()
        self.set_steps_to_run()

    def set_rank(self, rank):
        """Sets the rank of the current process."""
        # self.rank = None
        # raise NotImplementedError
        # To be implemented later
        self.rank = rank
        self.dataset_setup()
        if self.rank is not None:
            self.labeled_criterion = nn.BCELoss(reduction='none')
        elif self.rank is None:
            self.labeled_criterion = nn.BCELoss()


    def feature_distance_loss(self, base_features, other_features, distance_function=None):
        """Calculate the loss based on the distance between feature vectors."""
        epsilon = 1e-5

        # print("base_features: ", base_features.shape)
        # print("other_features: ", other_features.shape)
        if distance_function is None:
            distance_function = self.settings.matching_distance_function

        ### Come back to this later
        if self.settings.use_feature_covariance and not self.settings.use_feature_angle:
            if self.settings.normalize_feature_norm:
                base_features = base_features / (base_features.norm() + epsilon)
                other_features = other_features / (other_features.norm() + epsilon)
            # print("base_features: ", base_features.norm())
            #
            ## need to find a way to remove Nan values
            base_corrcoef = feature_corrcoef(base_features)
            other_corrcoef = feature_corrcoef(other_features)
            print("base_corrcoef: ", base_corrcoef)
            print("other_corrcoef: ", other_corrcoef)

            distance_vector = distance_function(base_corrcoef - other_corrcoef)
            print(distance_vector)
            # return distance_vector
        elif self.settings.use_feature_angle and not self.settings.use_feature_covariance:
            if self.settings.normalize_feature_norm:
                base_features = base_features / (base_features.norm() + epsilon)
                other_features = other_features / (other_features.norm() + epsilon)

            distance_vector =  feature_angle_loss(base_features, other_features, 
                                      summary_writer=self.gan_summary_writer,
                                        distance_function=distance_function)
            # print("distance_vector: ", distance_vector)
            return distance_vector
            # print("feature_covar: ", feature_covar)
            # return feature_covar
        base_mean_features = base_features.mean(0)
        other_mean_features = other_features.mean(0)
        if self.settings.normalize_feature_norm:
            base_mean_features = base_mean_features / (base_mean_features.norm() + epsilon)
            other_mean_features = other_features / (other_mean_features.norm() + epsilon)
        distance_vector = distance_function(base_mean_features - other_mean_features)
        # print("distance_vector: ", distance_vector)
        return distance_vector
    
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
        self.set_summary_step()
        self.set_steps_to_run()

    def dnn_training_step(self, examples, labels, step):
        """Runs an individual round of DNN training."""
        self.DNN.apply(disable_batch_norm_updates)  # No batch norm
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

    def labeled_loss_function(self, predicted_labels, labels, order):
        """Calculate the loss from the label difference prediction."""
        if self.rank is not None:
            y, r = labels
            labeled_loss = self.labeled_criterion(predicted_labels, y)
            labeled_loss = torch.mean(labeled_loss * r)
        else:
            labeled_loss = self.labeled_criterion(predicted_labels, labels)
        print("Labeled loss: ", labeled_loss.detach().numpy(),end='\r')
        return labeled_loss
            
    def labeled_loss_calculation(self, labeled_examples, labels):
        """Calculates the labeled loss."""
        predicted_labels = self.D(labeled_examples)
        self.labeled_features = self.D.features
        labeled_loss = self.labeled_loss_function(predicted_labels, labels, order=self.settings.labeled_loss_order)
        labeled_loss *= self.settings.labeled_loss_multiplier

        labeled_loss *= self.settings.srgan_loss_multiplier
        return labeled_loss
    
    def dnn_loss_calculation(self, labeled_examples, labels):
        """Calculates the DNN loss."""
        predicted_labels = self.DNN(labeled_examples)
        labeled_loss = self.labeled_loss_function(predicted_labels, labels, order=self.settings.labeled_loss_order)
        labeled_loss *= self.settings.labeled_loss_multiplier

        labeled_loss *= self.settings.dnn_loss_multiplier
        return labeled_loss

    def gan_training_step(self, labeled_examples, labels, unlabeled_examples, step):
        """Runs an individual round of GAN training."""
        # Labeled.
        self.D.apply(disable_batch_norm_updates)  # No batch norm
        self.gan_summary_writer.step = step
        self.d_optimizer.zero_grad()
        labeled_loss = self.labeled_loss_calculation(labeled_examples, labels)
        labeled_loss.backward()
        # Unlabeled.
        self.D.apply(disable_batch_norm_updates)  # Make sure only labeled data is used for batch norm statistics
        unlabeled_loss = self.unlabeled_loss_calculation(labeled_examples, unlabeled_examples)
        unlabeled_loss.backward()
        # Fake.
        z = torch.tensor(MixtureModel([norm(-self.settings.mean_offset, 1),
                                       norm(self.settings.mean_offset, 1)]
                                      ).rvs(size=[unlabeled_examples.size(0),
                                                  self.G.input_size]).astype(np.float32)).to(gpu)
        fake_examples = self.G(z)
        # print("Fake Examples: ", fake_examples)
        fake_loss = self.fake_loss_calculation(unlabeled_examples, fake_examples)
        fake_loss.backward()
        # Gradient penalty.
        gradient_penalty = self.gradient_penalty_calculation(fake_examples, unlabeled_examples)
        print("Gradient penalty: ", gradient_penalty.detach().numpy())
        gradient_penalty.backward()
        # Discriminator update.
        self.d_optimizer.step()
        # Generator.
        if step % self.settings.generator_training_step_period == 0:
            self.g_optimizer.zero_grad()
            z = torch.randn(unlabeled_examples.size(0), self.G.input_size).to(gpu)
            fake_examples = self.G(z)
            generator_loss = self.generator_loss_calculation(fake_examples, unlabeled_examples)
            generator_loss.backward()
            self.g_optimizer.step()
            if self.gan_summary_writer.is_summary_step():
                self.gan_summary_writer.add_scalar('Generator/Loss', generator_loss.item())
        # Summaries.
        if self.gan_summary_writer.is_summary_step():
            self.gan_summary_writer.add_scalar('Discriminator/Labeled Loss', labeled_loss.item())
            self.gan_summary_writer.add_scalar('Discriminator/Unlabeled Loss', unlabeled_loss.item())
            self.gan_summary_writer.add_scalar('Discriminator/Fake Loss', fake_loss.item())
            self.gan_summary_writer.add_scalar('Discriminator/Gradient Penalty', gradient_penalty.item())
            self.gan_summary_writer.add_scalar('Discriminator/Gradient Norm', self.gradient_norm.mean().item())
            if self.labeled_features is not None and self.unlabeled_features is not None:
                self.gan_summary_writer.add_scalar('Feature Norm/Labeled',
                                                   self.labeled_features.mean(0).norm().item())
                self.gan_summary_writer.add_scalar('Feature Norm/Unlabeled',
                                                   self.unlabeled_features.mean(0).norm().item())
                self.gan_summary_writer.add_image('Feature Corr/Labeled',
                                                    summwriter_feature_plot(self.labeled_features), step)
                self.gan_summary_writer.add_image('Feature Corr/Unlabeled',
                                                    summwriter_feature_plot(self.unlabeled_features), step)
                self.gan_summary_writer.add_image('Feature Corr/ Fake',
                                                    summwriter_feature_plot(self.fake_features), step)



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
    

    def predict_activiity(self, fp):
        """Predicts the activity of a given fingerprint."""
        ## Not tested
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

    def compute_MCC(self, predictions, labels):
        """Computes the Matthews Correlation Coefficient (MCC) for the given predictions and labels."""
        # Convert predictions to binary values (0 or 1)
        binary_predictions = torch.round(predictions)

        # Compute true positives, true negatives, false positives, and false negatives
        tp = torch.sum((binary_predictions == 1) & (labels == 1))
        tn = torch.sum((binary_predictions == 0) & (labels == 0))
        fp = torch.sum((binary_predictions == 1) & (labels == 0))
        fn = torch.sum((binary_predictions == 0) & (labels == 1))

        # Compute numerator and denominator of MCC equation
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        # Check for division by zero
        if denominator == 0:
            return 0.01

        # Compute MCC
        mcc = numerator / denominator

        return mcc


    def validation_summaries(self, step: int):
        settings = self.settings
        dnn_summary_writer = self.dnn_summary_writer
        gan_summary_writer = self.gan_summary_writer
        DNN = self.DNN
        D = self.D
        G = self.G
        train_dataset = self.train_dataset
        validation_dataset = self.validation_dataset
        unlabeled_dataset = self.unlabeled_dataset


        dnn_train_values = self.evaluation_epoch(DNN, train_dataset, dnn_summary_writer, '2 Train Error')
        dnn_validation_values = self.evaluation_epoch(DNN, validation_dataset, dnn_summary_writer, '1 Validation Error')
        gan_train_values = self.evaluation_epoch(D, train_dataset, gan_summary_writer, '2 Train Error')
        gan_validation_values = self.evaluation_epoch(D, validation_dataset, gan_summary_writer, '1 Validation Error',
                              comparison_values=dnn_validation_values)
        dnn_test_values = self.evaluation_epoch(DNN, self.test_dataset, dnn_summary_writer, '3 Test Error')
        gan_test_values = self.evaluation_epoch(D, self.test_dataset, gan_summary_writer, '3 Test Error')
        # Just so we are actually using the values
        print("DNN Test Values: ", dnn_test_values, end="\r")
        print("GAN Test Values: ", gan_test_values, end="\r")
        z = torch.tensor(MixtureModel([norm(-settings.mean_offset, 1), norm(settings.mean_offset, 1)]).rvs(
            size=[settings.batch_size, G.input_size]).astype(np.float32)).to(gpu)
        
        fake_examples = G(z)
        fake_examples_array = fake_examples.to('cpu').detach().numpy()
        fake_predicted_labels = D(fake_examples)
        fake_predicted_labels_array = fake_predicted_labels.to('cpu').detach().numpy()
        unlabeled_labels_array = unlabeled_dataset.LABELS[:len(self.validation_dataset)]
        label_wasserstein_distance = wasserstein_distance(fake_predicted_labels_array, unlabeled_labels_array)
        gan_summary_writer.add_scalar('Generator/Predicted Label Wasserstein', label_wasserstein_distance, step)

        unlabeled_examples_array = unlabeled_dataset.FP[:len(self.validation_dataset)]

        # unlabeled_examples = torch.tensor(unlabeled_examples_array.astype(np.float32)).to(gpu)
        unlabeled_examples = unlabeled_examples_array.to(gpu)

        unlabeled_predictions = D(unlabeled_examples)
        if dnn_summary_writer.step % settings.summary_step_period == 0:
            unlabeled_predictions_array = unlabeled_predictions.to('cpu').detach().numpy()
            validation_predictions_array = gan_validation_values.predicted_labels
            train_predictions_array = gan_train_values.predicted_labels
            dnn_validation_predictions_array = dnn_validation_values.predicted_labels
            dnn_train_predictions_array = dnn_train_values.predicted_labels
            distribution_image = generate_display_frame(fake_examples_array, unlabeled_predictions_array,
                                                        validation_predictions_array, dnn_validation_predictions_array,
                                                        train_predictions_array, dnn_train_predictions_array, step)
            distribution_image = standard_image_format_to_tensorboard_image_format(distribution_image)
            gan_summary_writer.add_image('Distributions', distribution_image)

    class ComparisonValues(RecordClass):
        """A record class to hold the names of values which might be compared among methods."""
        ACC: float
        BAC: float
        MCC: float
        predicted_labels: np.ndarray

    def evaluation_epoch(self, network, dataset: QSARDataset, summary_writer:SummaryWriter, summary_name: str,
                         comparison_values: ComparisonValues = None, step: int = None):
        """An evaluation of the dataset writing to TensorBoard."""

        if dataset is self.validation_dataset:
            dataloader = self.validation_dataset_loader

        elif dataset is self.train_dataset:
            dataloader = self.train_dataset_loader
        elif dataset is self.test_dataset:
            dataloader = self.test_dataset_loader


        self.eval_mode()

        #create list of labels from test dataset indexes
        labels = []
        predictions = []
        for data in dataloader:
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
        predicted_labels = torch.tensor(predictions)
        labels = torch.tensor(labels)

        # print("Predicted size: ", predicted_labels.shape)
        # print("Predicted Labels: ", predicted_labels)



        acc = self.compute_ACC(predicted_labels, labels)
        summary_writer.add_scalar(f'{summary_name}/ACC', acc, step)
        bac = self.compute_BAC(predicted_labels, labels)
        summary_writer.add_scalar(f'{summary_name}/BAC', bac, step)
        mcc = self.compute_MCC(predicted_labels, labels)
        summary_writer.add_scalar(f'{summary_name}/MCC', mcc, step)

        # if comparison_values is not None:
        #     summary_writer.add_scalar(f'{summary_name}/ Ratio ACC GAN/DNN', acc / comparison_values.ACC, step)
        #     summary_writer.add_scalar(f'{summary_name}/ Ratio BAC GAN/DNN', bac / comparison_values.BAC, step)
        #     summary_writer.add_scalar(f'{summary_name}/ Ratio MCC GAN/DNN', mcc / comparison_values.MCC, step)
        return self.ComparisonValues(ACC=acc, BAC=bac, MCC=mcc, predicted_labels=predicted_labels)
