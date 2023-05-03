import random
from collections import defaultdict
import numpy as np
from scipy.stats import norm

import pandas as pd
import os
from torch.utils.data import DataLoader
import torch
from torch import nn
import math
import datetime

### move these over to the SR-GAN.DomAdpQSAR folder to setup SRGAN model
from DomAdpQSAR.models import Classifier, Generator
from DomAdpQSAR.data_test_utils import QSARDataset, dataset_compiler

from DomAdpQSAR.srgan import feature_corrcoef

# import DomApdQSAR functions here

from DomAdpQSAR.utility import gpu, seed_all, make_directory_name_unique,MixtureModel


from DomAdpQSAR.QSARdnn import DomAdpQSARDNN

from DomAdpQSAR.srgan import Experiment, disable_batch_norm_updates


class DomAdpQSARSRGAN(DomAdpQSARDNN, Experiment):
    def __init__(self, settings):
        super(DomAdpQSARSRGAN, self).__init__(settings)
        super(Experiment, self).__init__(settings)

        # super(DomAdpQSARSRGAN).__init__(settings)
        self.generator_layer_sizes = self.settings.generator_layer_sizes
        self.discriminator_layer_sizes = self.settings.discriminator_layer_sizes
        self.generator_features = None
        
    def model_setup(self):
        self.DNN = Classifier(self.layer_sizes)
        self.D = Classifier(self.discriminator_layer_sizes)
        self.G = Generator(self.generator_layer_sizes)
    
    def labeled_loss_calculation(self, labeled_examples, labels):
        """Calculates the labeled loss."""
        predicted_logits = self.D(labeled_examples)
        if self.rank is not None:
            y, r = labels
            labeled_loss = self.labeled_criterion(predicted_logits, y)
            labeled_loss = torch.mean(labeled_loss * r)
        else:
            labeled_loss = self.labeled_criterion(predicted_logits, labels)

        labeled_loss *= self.settings.labeled_loss_multiplier
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
        # self.D.apply(disable_batch_norm_updates)  # Make sure only labeled data is used for batch norm statistics
        unlabeled_loss = self.unlabeled_loss_calculation(labeled_examples, unlabeled_examples)
        unlabeled_loss.backward()
        # Fake.
        z = torch.tensor(MixtureModel([norm(-self.settings.mean_offset, 1),
                                       norm(self.settings.mean_offset, 1)]
                                      ).rvs(size=[unlabeled_examples.size(0),
                                                  self.G.input_size]).astype(np.float32)).to(gpu)
        fake_examples = self.G(z)
        self.generator_features = self.G.features
        fake_loss = self.fake_loss_calculation(unlabeled_examples, fake_examples)
        fake_loss.backward()
        # Gradient penalty.
        gradient_penalty = self.gradient_penalty_calculation(fake_examples, unlabeled_examples)
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

            self.gan_summary_writer.add_image('D Feature Corr/Labeled', 
                                              feature_corrcoef(self.labeled_features),
                                               step)

            self.gan_summary_writer.add_image('D Feature Corr/UnLabeled', 
                                               feature_corrcoef(self.unlabeled_features),
                                               step)

            self.gan_summary_writer.add_image('D Feature Corr/Fake', 
                                               feature_corrcoef(self.generator_features),
                                               step)

            self.gan_summary_writer.add_image('G Feature Corr/UnLabeled', 
                                               feature_corrcoef(self.unlabeled_features),
                                               step)
                


            if self.labeled_features is not None and self.unlabeled_features is not None:
                self.gan_summary_writer.add_scalar('Feature Norm/Labeled',
                                                   self.labeled_features.mean(0).norm().item())
                self.gan_summary_writer.add_scalar('Feature Norm/Unlabeled',
                                                   self.unlabeled_features.mean(0).norm().item())
        # self.D.apply(enable_batch_norm_updates)  # Only labeled data used for batch norm running statistics
