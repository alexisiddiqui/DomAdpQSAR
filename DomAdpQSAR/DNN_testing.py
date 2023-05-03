# %%
### Notebook to test DNN Class|
from utility import gpu, seed_all

from QSARsettings import Settings

# %%
from QSARdnn import DomAdpQSARDNN


# %%
test_settings = Settings()
test_settings.trial_name = "test"
test_Experiment = DomAdpQSARDNN(test_settings)

# %%
test_Experiment.model_setup()

# %% [markdown]
# 

# %% [markdown]
# 

# %%
test_Experiment.dataset_setup()

# %%
test_Experiment.train_dataset_loader = test_Experiment.clean_dataset_loader

# %%
device = 'cpu'
print(test_settings.summary_step_period)

# %%
test_Experiment.train()

# %%



