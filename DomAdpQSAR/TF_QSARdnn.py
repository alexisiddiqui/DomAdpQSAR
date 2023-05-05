
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
from DomAdpQSAR.models import TF_Classifier, freeze_layers
from DomAdpQSAR.data_test_utils import QSARDataset, dataset_compiler

from DomAdpQSAR.srgan import feature_corrcoef

# import DomApdQSAR functions here

from DomAdpQSAR.utility import gpu, seed_all, make_directory_name_unique


# from srgan import Experiment

# from QSARsettings import Settings

from DomAdpQSAR.QSARdnn import DomAdpQSARDNN

from DomAdpQSAR.QSARsettings import Settings
settings = Settings()
model = DomAdpQSARDNN(settings)
model.settings.load_model_path = "logs/Broad Tuning Underfit/"
model.model_setup()
featuriser = model.DNN

class TF_DomAdpQSARDNN(DomAdpQSARDNN):
    def model_setup(self):
        """Sets up the model."""
        self.DNN = TF_Classifier(self.settings.transfer_layer_sizes, featuriser=featuriser)
        self.freeze_DNN_layers()