#################################################################################################################################
##### Modules ###################################################################################################################
#################################################################################################################################

from torch.nn.utils import rnn
from calendar import month_abbr
import os
from queue import Full 
import random
import numpy as np
from typing import Dict
import pandas as pd
from pytorch_lightning.loggers import NeptuneLogger


pd.options.mode.chained_assignment = None  # default='warn'
import time

# Ignore warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning) 

from termcolor import colored # Colored output in terminal
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import RobustScaler

from torch.utils.data import TensorDataset, DataLoader
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, train_test_split, TimeSeriesSplit, cross_val_score




# Setting parent directory
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 


##### Importing supporting scripts #######
 
# Import analysis.py 
from analysis import sns_lineplot
# Importing utilities.py
from utilities import BlockingTimeSeriesSplit, get_cv_type, inverse_scale_features, scale_features, train__val_test_split, train_val_test_split_joined, ordinal_encoding, time_difference, generate_time_lags, robust_scaler, remove_nans, remove_nans_cv, get_first_non_na_row


sns.set(rc={'axes.facecolor':'white', 'axes.edgecolor':'black', 'figure.facecolor':'white'})

from pytorch_lightning import LightningModule, LightningDataModule
import pytorch_lightning as pl
from pytorch_forecasting.data.encoders import TorchNormalizer, NaNLabelEncoder
from pytorch_forecasting import LSTM, TimeSeriesDataSet, TemporalFusionTransformer, metrics
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.models import AutoRegressiveBaseModel, BaseModel



import torch
torch.cuda.is_available()

if torch.cuda.is_available():
        gpus = 1
else:
    gpus = 0
from model_definitions import * 



        


class LSTMModel(AutoRegressiveBaseModel):

    """
    LSTTM with 
    """
    def __init__(
        self,
        target: str,
        target_lags: Dict[str, Dict[str, int]],
        n_layers: int,
        hidden_size: int,
        dropout: float = 0,
        **kwargs,
    ):
        # hpparams saves hyperparameters
        self.save_hyperparameters()
        
        super().__init__(**kwargs)
        self.lstm = LSTM(
            hidden_size=self.hparams.hidden_size,
            input_size=1,
            num_layers=self.hparams.n_layers,
            dropout=self.hparams.dropout,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.hparams.hidden_size, 1)

    def encode(self, x: Dict[str, torch.Tensor]):
        assert x["encoder_lengths"].min() >= 1
        input_vector = x["encoder_cont"].clone()
        # lag target by one
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        input_vector = input_vector[:, 1:]  # first time step cannot be used because of lagging

        # determine effective encoder_length length
        effective_encoder_lengths = x["encoder_lengths"] - 1
        # run through LSTM network
        _, hidden_state = self.lstm(
            input_vector, lengths=effective_encoder_lengths, enforce_sorted=False  # passing the lengths directly
        )  # second ouput is not needed (hidden state)
        return hidden_state

    def decode(self, x: Dict[str, torch.Tensor], hidden_state):
        # again lag target by one
        input_vector = x["decoder_cont"].clone()
        input_vector[..., self.target_positions] = torch.roll(
            input_vector[..., self.target_positions], shifts=1, dims=1
        )
        # but this time fill in missing target from encoder_cont at the first time step instead of throwing it away
        last_encoder_target = x["encoder_cont"][
            torch.arange(x["encoder_cont"].size(0), device=x["encoder_cont"].device),
            x["encoder_lengths"] - 1,
            self.target_positions.unsqueeze(-1),
        ].T
        input_vector[:, 0, self.target_positions] = last_encoder_target

        if self.training:  # training mode
            lstm_output, _ = self.lstm(input_vector, hidden_state, lengths=x["decoder_lengths"], enforce_sorted=False)

            # transform into right shape
            prediction = self.output_layer(lstm_output)
            prediction = self.transform_output(prediction, target_scale=x["target_scale"])

            # predictions are not yet rescaled
            return prediction

        else:  # prediction mode
            target_pos = self.target_positions

            def decode_one(idx, lagged_targets, hidden_state):
                x = input_vector[:, [idx]]
                # overwrite at target positions
                x[:, 0, target_pos] = lagged_targets[-1]  # take most recent target (i.e. lag=1)
                lstm_output, hidden_state = self.lstm(x, hidden_state)
                # transform into right shape
                prediction = self.output_layer(lstm_output)[:, 0]  # take first timestep
                return prediction, hidden_state

            # make predictions which are fed into next step
            output = self.decode_autoregressive(
                decode_one,
                first_target=input_vector[:, 0, target_pos],
                first_hidden_state=hidden_state,
                target_scale=x["target_scale"],
                n_decoder_steps=input_vector.size(1),
            )

            # predictions are already rescaled
            return output

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hidden_state = self.encode(x)  # encode to hidden state
        output = self.decode(x, hidden_state)  # decode leveraging hidden state

        return self.to_network_output(prediction=output)


class FullyConnectedModule(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int):
        super().__init__() # call to superclass
        module_list = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        # Adding hidden layers
        for _ in range(n_hidden_layers):
            module_list.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        # output layer
        module_list.append(nn.Linear(hidden_size, output_size))

        self.sequential = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)

class FullyConnectedModel(BaseModel):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, n_hidden_layers: int, **kwargs):
        # saves arguments hparams
        self.save_hyperparameters()
        # pass additional arguments to BaseModel.__init__, mandatory call - do not skip this
        super().__init__(**kwargs)
        self.network = FullyConnectedModule(
            input_size=self.hparams.input_size,
            output_size=self.hparams.output_size,
            hidden_size=self.hparams.hidden_size,
            n_hidden_layers=self.hparams.n_hidden_layers,
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # x is a batch generated based on the TimeSeriesDataset
        network_input = x["encoder_cont"].squeeze(-1)
        prediction = self.network(network_input).unsqueeze(-1)
        prediction = self.transform_output(prediction, target_scale=x["target_scale"]) # Resclae prodictions

        # Dictionary that contains the prediction
        # The conversion to a named tuple can be directly achieved with the `to_network_output` function.
        return self.to_network_output(prediction=prediction)

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataSet, **kwargs):
        new_kwargs = {
            "output_size": dataset.max_prediction_length,
            "input_size": dataset.max_encoder_length,
        }
        new_kwargs.update(kwargs)  # use to pass real hyperparameters and override defaults set by dataset
        # example for dataset validation
        assert dataset.max_prediction_length == dataset.min_prediction_length, "Decoder only supports a fixed length"
        assert dataset.min_encoder_length == dataset.max_encoder_length, "Encoder only supports a fixed length"
        assert (
            len(dataset.time_varying_known_categoricals) == 0
            and len(dataset.time_varying_known_reals) == 0
            and len(dataset.time_varying_unknown_categoricals) == 0
            and len(dataset.static_categoricals) == 0
            and len(dataset.static_reals) == 0
            and len(dataset.time_varying_unknown_reals) == 1
            and dataset.time_varying_unknown_reals[0] == dataset.target
        ), "Only covariate should be the target in 'time_varying_unknown_reals'"

        return super().from_dataset(dataset, **new_kwargs)




###################################################

# Test using univariate Pytorch 


timeseries_kwargs = {'time_idx': 'seq',
                     'target':'dayahead_price', 
                     'min_encoder_length':10,
                     'max_encoder_length':10,
                     'min_prediction_length':2,
                     'max_prediction_length':2,
                     'group_ids': ['time_groups'],
                     'time_varying_unknown_reals':['dayahead_price']}


data_set = TimeSeriesDataSet(data  = small_df, **timeseries_kwargs)

    
data_module = DataModuleTimeSeries(small_df, batch_size= 20, num_workers= 12,  **timeseries_kwargs)



neptune_logger = NeptuneLogger(
        project="MasterThesis/Master-thesis",
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY3M2E4ZC0xYjIzLTQ5OWYtYTA0MC04NmVjOWRkZWViNmIifQ==")


################################################
# LSTM nn.modukle (non-pytorch lightning)

#fcm = FullyConnectedModel(input_size=5, output_size=2, hidden_size=10, n_hidden_layers=2)
fcm = FullyConnectedModel.from_dataset(data_set,  input_size=10, output_size=2, hidden_size=10, n_hidden_layers=2)

# Fitting fully connected model
trainer = pl.Trainer(
        max_epochs                  = 30,
        gpus                        = 1,
        num_processes= 0, 
        gradient_clip_val           = 0.1,
        logger= neptune_logger,
        limit_train_batches         = 30
    )

trainer.fit(fcm,  datamodule = data_module)


trainer.predict(fcm, data_module.predict_dataloader())


tests = trainer.test(fcm, data_module.val_dataloader())[0]

# Predictions with fully connected feedforward network

preds = fcm.predict(data_set, mode  = 'raw')
preds_as_numpy = preds[0].cpu().detach().numpy().flatten()




################################################
# LSTM nn.modukle (non-pytorch lightning)


#lstm_model = LSTMModel('intraday_price', target_lags = {}, hidden_size=2, n_layers = 2) 
lstm_model = LSTMModel.from_dataset(data_set, n_layers = 1, hidden_size = 1)
lstm_model.summarize('full')
lstm_model.hparams




trainer = pl.Trainer(
        max_epochs                  = 30,
        gpus                        = 0,
        gradient_clip_val           = 0.1,
        limit_train_batches         = 30,
        logger = neptune_logger
    )


#trainer.fit(lstm_model,  data_set.to_dataloader(train = True, shuffle = False, batch_size=20))
trainer.fit(lstm_model, datamodule = data_module)







if __name__ == '__main__':
    freeze_support()
    print("Do somethin")
    #train_and_validate_model(small_df, lstm_model, timeseries_kwargs, model_configuration)