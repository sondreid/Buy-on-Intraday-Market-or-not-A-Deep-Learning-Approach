"""


Common functions and
utilities for forecasting models


"""


#################################################################################################################################
##### Modules ###################################################################################################################
#################################################################################################################################

import os
from sqlite3 import Timestamp

from pytorch_lightning.loggers import NeptuneLogger
from neptune.new.types import File
import flash
from pytorch_forecasting.metrics import MultiHorizonMetric, TorchMetricWrapper
from typing import Dict, Optional
import itertools
import math
import copy
import sys
import numpy as np
import pandas as pd
import neptune.new as neptune
import glob
import gc
pd.options.mode.chained_assignment = None  # default='warn'
import time

from statistics import mean
# Ignore warning
import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.simplefilter(action='ignore', category=FutureWarning) 

from termcolor import colored # Colored output in terminal
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import RobustScaler
from pytorch_lightning import LightningModule, LightningDataModule
from torch.utils.data import TensorDataset

from sklearn.model_selection import cross_validate, train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import torch.utils.data
from torch.utils.data import DataLoader



# Setting parent directory
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 


##### Importing supporting scripts #######
 
# Import analysis.py 
from analysis import sns_lineplot
# Importing utilities.py
#from utilities import BlockingTimeSeriesSplit, SlidingWindowSplitThreeFolds, get_cv_type, inverse_scale_features, scale_features, train__val_test_split, train_val_test_split_joined, ordinal_encoding, time_difference, generate_time_lags, remove_nans, remove_nans_cv, get_first_non_na_row, shift_covariates
from utilities import *

sns.set(rc={'axes.facecolor':'white', 'axes.edgecolor':'black', 'figure.facecolor':'white'})

from pytorch_lightning import LightningModule
from pytorch_lightning.plugins.training_type import DDPPlugin
import pytorch_lightning as pl
from pytorch_forecasting.data.encoders import TorchNormalizer, NaNLabelEncoder
from pytorch_forecasting import BaseModel, DeepAR, TimeSeriesDataSet, RecurrentNetwork, metrics, TemporalFusionTransformer, BaseModelWithCovariates
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, StochasticWeightAveraging, ModelCheckpoint
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.metrics import MAE, MASE, MAPE, SMAPE, RMSE, QuantileLoss
from torch.nn import HuberLoss
# To combat tensorflow-api error
import tensorflow as tf

import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
from pytorch_lightning.loggers import TensorBoardLogger

import torch
torch.cuda.is_available()

if torch.cuda.is_available():
    gpus = 1
else:
    gpus = 0

selected_series =  pd.read_csv("../model_data/selected_series.csv")
#selected_series.seq                 = selected_series.seq.astype(int)
selected_series.nordlink            = selected_series.nordlink.astype(str)
selected_series.covid               = selected_series.covid.astype(str)
selected_series.northsealink        = selected_series.northsealink.astype(str)
selected_series.hour                = selected_series.hour.astype(str)
selected_series.day                 = selected_series.day.astype(str)
selected_series.year                = selected_series.year.astype(str)
selected_series.week                = selected_series.week.astype(str)
selected_series.month               = selected_series.month.astype(str)

selected_series = selected_series.drop(columns = ['Unnamed: 0'])


# Creating a small subset of data for testing purposes
series_df_small                    = selected_series[selected_series['datetime'] < '2021-06-01']
series_df_small['seq']             = np.arange(0, len(series_df_small)).tolist()
series_df_small.reset_index(drop   = True, inplace = True)
series_df_small['time_groups']     = np.repeat("group1", len(series_df_small))
series_df_small.time_groups        = series_df_small.time_groups.astype(str)



# Format entire set
series_df                = selected_series
series_df['seq']         = np.arange(0, len(series_df)).tolist()
series_df.reset_index(drop = True, inplace = True)
series_df['time_groups'] = np.repeat("group1", len(series_df))
series_df.time_groups         = series_df.time_groups.astype(str)





#TODO expand evaluation period
series_df_evaluation = series_df[series_df['datetime'] >= '2022-01-01']
series_df            = series_df[series_df['datetime'] < '2022-01-01']



################################# Definining features/variables ####################################
# Include all lags

lag_vars =     ['intraday_price_lag1','intraday_price_lag2','intraday_price_lag4','intraday_price_lag13','intraday_price_lag17',
                'intraday_price_lag21', 'intraday_price_lag23', 'intraday_price_lag32', 'intraday_price_lag45',
                'dayahead_price_lag1', 'dayahead_price_lag2','dayahead_price_lag3','dayahead_price_lag9','dayahead_price_lag13',
                'dayahead_price_lag16','dayahead_price_lag22','dayahead_price_lag23','dayahead_price_lag25','dayahead_price_lag46']


# Specifications for TimeSeriesDataset


forecast_vars = [covariate for covariate in series_df.columns if "Forecast" in covariate or 'REMIT' in covariate]

capacity_vars = [covariate for covariate in series_df.columns if "Day-Ahead Capacity" in covariate]


time_varying_known_categoricals         = ['day', 'hour', 'year', 'week', 'month', 'nordlink', 'northsealink']
time_varying_unknown_categoricals       = ['covid']

non_forecast_vars = [covariate for covariate in series_df.columns if covariate not in forecast_vars + capacity_vars + time_varying_known_categoricals + time_varying_unknown_categoricals + ['seq']+['datetime']+['time_groups']]
non_forecast_vars =[covariate for covariate in non_forecast_vars if 'price_lag' not in covariate]
time_varying_known_reals                = forecast_vars + capacity_vars
time_varying_unknown_reals              =  non_forecast_vars + lag_vars





#TODO When calculating generalization error, this needs change
# Set Cross validation type

cv = get_cv_type('sliding', 5, **{'val_size':240})


max_prediction_length  = 38


# Shifting the forecast h periods back in time in order to handle forecasted values
series_df_shifted = series_df.copy()
series_df_shifted[time_varying_known_reals] = shift_covariates(
    df      = series_df_shifted, 
    columns = time_varying_known_reals, 
    h       = -max_prediction_length  
    )
    







##### Data module ########################################################################################################

        
def scale_remove_nans(df, target):
    robust_scaler  = RobustScaler()
    df_no_nans = remove_nans(df, 'linear')
    return scale_features(df_no_nans, target, robust_scaler)

class PandasDataset(Dataset):
    """
    Class for converting pandas dataframe to 
    Pytorch Dataset
    """
    def __init__(self, df, target):
        X = df.copy().drop(columns = [target])
        self.X = torch.from_numpy(X.to_numpy())
        y =  (df[target].to_numpy() > 0).astype(int)
        print(" > Len y {} target {}".format(len(y), len(target)))
        assert len(y) == len(df[target]), "Output not similar length"
        
        self.y = torch.from_numpy(y).float()

        

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
      return self.X[i], self.y[i]

    def data(self):
        return self.X

    def X(self):
        return self.X

    def y(self):
        return self.y


class DifferenceDataModule(LightningDataModule):

    """
    Data module inherited from Pytorch lightning data module.

    Superclass LightningDataModule
    """


    def __init__(self, df, train_indices, val_indices,  target, batch_size, num_workers, **kwargs):    
        """
        Constructor method
        """

        self.drop_last = False
        if kwargs.get('drop_last'):
            print(colored("> Drop last", 'green'))
            self.drop_last = True
        self.scale = True
        if kwargs.get('scale') is not None:
            self.scale = kwargs.get('scale') 
        super().__init__()
        self.df = df
        self.target  = target

        # Batch size and number of workers
        self.batch_size             = batch_size
        self.num_workers            = num_workers
        for col in self.df.columns: # Setting types
            self.df[col] = self.df[col].astype(np.float32)
        """        train, val  = train_test_split(
                                self.df, 
                                test_size    = 0.15,      # Validation split
                                shuffle      = False,      
                                random_state = 144)
        """
        train = self.df.iloc[train_indices]
        val   = self.df.iloc[val_indices]


        # Removing missing values with linear interpolation
        train = scale_remove_nans(train, target)
        val   = scale_remove_nans(val, target)


        self.train = PandasDataset(train, target = target)
        self.val   = PandasDataset(val, target = target)

        
        

    # Dataloader class methods
    def train_dataloader(self):
        try:
            return DataLoader(self.train, batch_size=self.batch_size, shuffle = False,  drop_last = self.drop_last, pin_memory=True, num_workers=self.num_workers)
    
        except:
            raise ValueError("> No train timeseries has been set")
    
    def val_dataloader(self):
        try:
            return DataLoader(self.val, batch_size=self.batch_size, shuffle = False, drop_last = self.drop_last, pin_memory=True, num_workers=self.num_workers)
        except:
            raise ValueError("> No val timeseries has been set")

    def test_dataloader(self):
        try:
            return DataLoader(self.test, batch_size=self.batch_size, shuffle = False, drop_last = self.drop_last, pin_memory=True, num_workers=self.num_workers)
        except:
            raise ValueError("> No test timeseries has been set")
    def predict_dataloader(self):
        return self.test_dataloader()

    def set_test_dataloader(self, test_df):
        """
        Setting test dataloader after data module initialisation
        """
        for col in test_df.columns: # Setting types
            test_df[col] = test_df[col].astype(np.float32)
        test_df[self.target] = test_df[self.target].astype(np.float32)
        test = scale_remove_nans(test_df, self.target)
        self.test   = PandasDataset(test, self.target)
        




class DataModuleTimeSeries(LightningDataModule):

    """
    Data module inherited from Pytorch lightning data module.

    Superclass LightningDataModule
    """


    def __init__(self, df, train_indices = None, val_indices = None, test_indices = None,  **timeseries_kwargs):
        """
        Constructor method
        """
        super().__init__()
        self.target = timeseries_kwargs.get('target')
        # Batch size and number of workers
        # Remove from dictionary after retrieval
        self.batch_size = timeseries_kwargs.pop('batch_size')
        self.num_workers = timeseries_kwargs.pop('num_workers')
        scale = timeseries_kwargs.pop('scale')
        #print("> Time series kwargs", timeseries_kwargs)
        self.df = df
        self.timeseries_kwargs = timeseries_kwargs      
        

        self.robust_scaler = RobustScaler()
        # Converting the features to correct data types
        self.df.seq                 = self.df.seq.astype(int)
        self.df.nordlink            = self.df.nordlink.astype(str)
        self.df.covid               = self.df.covid.astype(str)
        self.df.northsealink        = self.df.northsealink.astype(str)
        self.df.time_groups         = self.df.time_groups.astype(str)
        self.df.hour                = self.df.hour.astype(str)
        self.df.day                 = self.df.day.astype(str)
        self.df.year                = self.df.year.astype(str)
        self.df.week                = self.df.week.astype(str)
        self.df.month               = self.df.month.astype(str)
                    
        # If no train, validation, test indides are specified --> perform simple static train, validation and split
        if train_indices is None or val_indices is None or test_indices is None:
                    train, val  = train_test_split(
                                    self.df, 
                                    test_size    = 0.15,    # Validation split
                                    shuffle      = False,  # For obvious reasons
                                    random_state = 144)

                    # Removing missing values with linear interpolation
                    train   = remove_nans(train, 'linear')
                    val     = remove_nans(val, 'linear')
                    
                    if scale:
                        train = scale_features(train, self.target, self.robust_scaler)
                        val   = scale_features(val, self.target, self.robust_scaler)

                    self.train_timeseries       = TimeSeriesDataSet(data = train, **timeseries_kwargs)
                    self.validation_timeseries  = TimeSeriesDataSet(data = val,   **timeseries_kwargs)


        else:
   
            train   = self.df.iloc[train_indices,:]
            val     = self.df.iloc[val_indices,:]
            test    = self.df.iloc[test_indices,:]
            # Removing missing values with linear interpolation
            train   = remove_nans(train, 'linear')
            val     = remove_nans(val,   'linear')
            test    = remove_nans(test, 'linear')

            if scale:
                train = scale_features(train, self.target, self.robust_scaler)
                val   = scale_features(val,   self.target, self.robust_scaler)
                test  = scale_features(test,  self.target, self.robust_scaler)

        
            self.train_timeseries       = TimeSeriesDataSet(data = train, **timeseries_kwargs)
            self.validation_timeseries  = TimeSeriesDataSet(data = val,   **timeseries_kwargs)

            
            # importing copy module
            import copy
            test_timeseries_kwargs = copy.deepcopy(timeseries_kwargs)
            test_timeseries_kwargs.pop('min_prediction_length')
            pred_length = test_timeseries_kwargs.pop('max_prediction_length')
            self.test_timeseries        = TimeSeriesDataSet(data = test, min_prediction_length  = pred_length, max_prediction_length = pred_length, predict_mode = True, **test_timeseries_kwargs)



    # Dataloader class methods
    def train_dataloader(self):
        try:
            return self.train_timeseries.to_dataloader(train = False, batch_size = self.batch_size, num_workers = self.num_workers, pin_memory=True)
        except:
            raise ValueError("> No train timeseries has been set")
    

    def val_dataloader(self):
        return self.validation_timeseries.to_dataloader(train = False, batch_size = self.timeseries_kwargs.get('max_prediction_length'), num_workers = self.num_workers, pin_memory=True)
        #return self.validation_timeseries.to_dataloader(train = False, batch_size =  50, num_workers = self.num_workers, pin_memory=True)

    def test_dataloader(self):
        try:
            return self.test_timeseries.to_dataloader(train = False, batch_size = self.timeseries_kwargs.get('max_prediction_length'), num_workers = self.num_workers, pin_memory=True)
            #return self.test_timeseries.to_dataloader(train = False, batch_size = self.timeseries_kwargs.get('max_prediction_length') +5, num_workers = self.num_workers, pin_memory=True)
        except:
            raise ValueError("> No test timeseries has been set")

    def predict_dataloader(self):
        return self.test_dataloader()

    def set_train_dataloader(self, train_df):
        """ Overrride train timeseries """
        train = remove_nans(train_df, 'linear')
        self.train_timeseries  = TimeSeriesDataSet(data = train, **self.timeseries_kwargs)


    def set_val_dataloader(self, val_df):
        """ Overrride validation timeseries """
        val = remove_nans(val_df, 'linear')
        self.validation_timeseries  = TimeSeriesDataSet(data = val, **self.timeseries_kwargs)

    def set_test_dataloader(self, test_df):
        test = remove_nans(test_df, 'linear')
        self.test_timeseries  = TimeSeriesDataSet(data = test, **self.timeseries_kwargs)





def train_model(
    data_module: DataModuleTimeSeries, 
    model: BaseModelWithCovariates,
    neptune_logger,
    model_configuration,
    gpu_bool: int = gpus,
    **kwargs) -> tuple:
   
    """
    Trains and fits a model based on input timeseries and dataloaders
   
    Parameters:
        logger: instantiated logger object. For instance a NeptuneLogger
    """
    print(colored("\n > Fitting model pending...", "blue"))

    if gpu_bool == 1:
        accelerator = 'gpu'
    else:
        accelerator = 'cpu'

    print(colored("\n > Model configurations" + str(model_configuration), "magenta"))
    if kwargs.get('verbose'):
        print(colored("\n > Initial parameters"   + str(model.hparams_initial), "magenta"))
    
    patience = model_configuration.get('patience')
    if patience is None:
        patience = 5


    gradient_clip_val = model_configuration.get('gradient_clip_val')
    if gradient_clip_val is None:
        gradient_clip_val = 0.1


    max_epochs = model_configuration.get('max_epochs')
    if max_epochs is None:
        print("> Epochs haven't been set")
        max_epochs = 20


    min_delta = model_configuration.get('min_delta')
    if min_delta is None:
        min_delta = 0.1

    print("> Min delta", min_delta)
    accumulate_grad_batches =  model_configuration.get('accumulate_grad_batches')
    if kwargs.get('accumulate_grad_batches') is None:
        accumulate_grad_batches =  1

    auto_scale_batch_size = model_configuration.get('auto_scale_batch_size')


    #######              Callbacks           ##############

    val_metric = model_configuration.get('val_metric')
    if val_metric is None:
        val_metric = 'val_MAE'

    patience_mode = model_configuration.get('patience_mode')
    if patience_mode is None:
        patience_mode = 'min'
    print(colored("> patience mode {}".format(patience_mode), 'green'))
    early_stop_callback = EarlyStopping(
        monitor                     = val_metric,  
        min_delta                   = min_delta, 
        patience                    = patience, 
        verbose                     = True, 
        mode                        = patience_mode
    )
    
    lr_logger = LearningRateMonitor(logging_interval='epoch')


    if model_configuration.get('k_best_checkpoints') is not None: save_top_k = model_configuration.get('k_best_checkpoints')
    else: save_top_k = 5


    if not os.path.exists("model_checkpoints"):
            os.makedirs("model_checkpoints")
    checkpoint_callback = ModelCheckpoint(dirpath='model_checkpoints/', 
                                filename = kwargs.get('model_name'),
                                monitor = val_metric, mode = 'min', 
                                save_top_k = save_top_k, save_weights_only=False, )
    callbacks                   = [early_stop_callback, lr_logger, checkpoint_callback]

    swa_epoch_start = model_configuration.get('swa_epoch_start')
    if model_configuration.get('swa_epoch_start') is None:
        swa_epoch_start = 15
    stoch_weight_avg = StochasticWeightAveraging(swa_epoch_start = swa_epoch_start, swa_lrs=1e-2) # Adding stochastic weight averaging: 
    
    callbacks.append(stoch_weight_avg)
    print(colored(" > CALLBACKS {}".format(callbacks), 'yellow'))


    if sys.platform == 'linux':
        strategy = None
        print("Strategy", strategy)
    else:
        strategy = "ddp-spawn"
    torch.cuda.empty_cache()

    #TODO Implement accumulate gradient schedule if necessary
    #from pytorch_lightning.callbacks import GradientAccumulationScheduler
    #accumulator = GradientAccumulationScheduler(scheduling={0: 8, 4: 4, 8: 1})
    """
    if type(model_configuration.get('accumulate_grad_batches')) is dict:
        accumulator = GradientAccumulationScheduler(scheduling=model_configuration.get('accumulate_grad_batches'))
        callbacks.append(accumulator)
    """


    ##### Logging ##### 
    if neptune_logger is not None:
        logger = neptune_logger
    else:
        tb_logger = TensorBoardLogger(save_dir="tb_logs", name="nameless")
        tb_logger.log_hyperparams(model.hparams)

        if kwargs.get('model_name') is not None:
            tb_logger = TensorBoardLogger(save_dir="tb_logs", name=kwargs.get('model_name'))
        tb_logger.log_hyperparams(model.hparams)
        tb_logger.log_hyperparams(model_configuration)
        logger = tb_logger




    trainer = pl.Trainer(
            max_epochs                  = max_epochs,
            gpus                        = gpu_bool,
            accelerator                 = accelerator,
            benchmark = False, 
            devices = 1,
            auto_scale_batch_size = auto_scale_batch_size,
            gradient_clip_val           = gradient_clip_val,
            #val_check_interval=1, 
            check_val_every_n_epoch=1,
            accumulate_grad_batches = accumulate_grad_batches, 
            logger = logger,
            callbacks                   = callbacks
        )
    
    trainer.fit(model, data_module)
    
    """    if neptune_logger is not None:
            neptune_logger['train/loss'] = trainer.logged_metrics"""

    print(colored("\n > Fitting model complete...", "green"))

    return trainer, model




#################################################################################################################################
##### Model Evaluation #########################################################################################################
#################################################################################################################################

def train_make_plot(actuals : pd.Series, forecasts : pd.Series, save : str, legend_text, neptune_logger = None):
    """
    For use during model training
    Saves plots in train_images directory
    
    """
    if type(actuals) != pd.Series:
        raise TypeError("> Actuals should be type pandas series")
    assert len(actuals) == len(forecasts.iloc[0:len(actuals)]), "Predictions and actuals are not of equal length"
    forecasts_and_actuals = pd.DataFrame({'seq' : np.arange(0, len(actuals.values)).tolist(), 'forecasts': forecasts.iloc[0:len(actuals)].values, 'actuals':actuals.values})

    forecasts_and_actuals_long = forecasts_and_actuals.melt(id_vars = ['seq'], 
                    var_name = 'type', 
                    value_name = 'euro_mwh', value_vars = ['forecasts', 'actuals'])

    if not os.path.exists("train_images"):
            os.makedirs("train_images")
    now = Timestamp.now()
    now = now.strftime('%Y-%m-%d %H:%M:%S')
    if not os.path.exists("train_images/benchmarks"):
            os.makedirs("train_images/benchmarks")
    directory = "train_images/"
        
    sns_lineplot(df = forecasts_and_actuals_long, x = forecasts_and_actuals_long.seq, y = forecasts_and_actuals_long.euro_mwh, xlab= "Hours", ylab = "Price €",  hue = 'type', save=True, directory= directory, legend_text = legend_text, file_title= now+save )
    
    
    """
    Neptune logger is of generic type and not of pytorch-lightning type (
        see documentation if unsure
    )
    
    """
    file_name = directory +now+save+".png"
    if neptune_logger is not None:
        neptune_logger["train/images"].log(File(file_name))
    return file_name


###########################################
#### Custom loss metrics  #####

class MSE(MultiHorizonMetric):
    def loss(self, y_pred, target):
        loss = (self.to_prediction(y_pred) - target)**2
        return loss


class HuberOld(MultiHorizonMetric):
    """
    Implements Huber loss metric
    https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html 
    """
    def loss(self, y_pred, target, delta = 1 ):
        mae = (self.to_prediction(y_pred) - target).abs()
        print(colored(">Mae {} type {}").format(mae, type(mae)))
        if mae < delta:
            mse =  (self.to_prediction(y_pred) - target)**2
            return 0.5*mse
        return delta * (mae - 0.5*delta)
  
class Huber(TorchMetricWrapper):

    def __init__(self, torchmetric, reduction: str = None, **kwargs):
        """
        Args:
            torchmetric (LightningMetric): Torchmetric to wrap.
            reduction (str, optional): use reduction with torchmetric directly. Defaults to None.
        """
        super().__init__(torchmetric = torchmetric, reduction=reduction, **kwargs)
      
def take_first_model(mode_config_dicts) -> dict:
    
    keys, values = zip(*mode_config_dicts.items())
    model_configurations_list = [dict(zip(keys, v)) for v in itertools.product(*values)] # All possible model configurations
    return model_configurations_list[0]


def run_specific_fold(df: pd.DataFrame, 
                     model: BaseModelWithCovariates,    
                     cv,
                     fold :int, 
                     model_name:str,
                     timeseries_kwargs: dict,
                     model_configuration: dict,            
                     verbose = False,
                     gpu_bool = gpus):
        """
        
        Run a specific CV fold (outer fold).
        Runs expanding CV on inner folds

        """
        # Specifify prediction length
        prediction_length = timeseries_kwargs.get('max_prediction_length')
        train_indices, val_indices, test_indices = [fold for fold in cv.split(df)][fold]
        complete_folds =  math.floor(len(val_indices)/prediction_length) # Complete validation folds
        print(colored("> Starting walk-forward forecasts on validation set", "cyan"))

        neptune_logger = NeptuneLogger(
            project="MasterThesis/Tweaking",
            api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY3M2E4ZC0xYjIzLTQ5OWYtYTA0MC04NmVjOWRkZWViNmIifQ==",
            log_model_checkpoints=False,
            source_files=["model_definitions.py", "run_lstm.py", "run_tft.py", "run_gru.py", "run_deepar.py", "model_tweaking.py"]
        )

        kwargs = {'model_name': model_name}

        neptune_logger.experiment['model_name'].log(kwargs.get('model_name'))
        #neptune_logger.experiment['target'].log(target)
            # Initial values and dictionary structures
        total_performance = {
            'mean_val_mae'  : [],
            'mean_val_mape' : [],
            'mean_val_smape' : [],
            'mean_val_rmse' : [],
            'mean_val_loss' : []
        }
        training_and_val_data    = {}

        for j in range(1, complete_folds+1):
            
            new_train_indices, new_val_indices, new_test_indices = non_overlapping_train_val_folds(j, prediction_length, train_indices, val_indices, test_indices)
            
            data_module = DataModuleTimeSeries(df = df, train_indices = new_train_indices, val_indices=new_val_indices, test_indices=new_test_indices, **timeseries_kwargs)
            
            if verbose : print(colored(">  Last train index {}".format(new_val_indices) ,'cyan')) 
            if verbose: print(colored(">  Val indices {}".format(new_val_indices) ,'cyan'))
            
        
            trainer, model = train_model(data_module=data_module, model= model, neptune_logger = neptune_logger, gpu_bool=gpu_bool,model_configuration=model_configuration, **kwargs)


            if model_configuration.get('use_best_checkpoint'): 
                try:
                    
                    model_call = model_configuration.get("model_call")
                    print(colored("> Model call {}".format(model_call), 'green'))
                
                except:
                    raise ValueError("Missing stored")
                print(colored("> Using best checkpoint....", 'green'))
                best_model_path = trainer.checkpoint_callback.best_model_path
                model = model_call.load_from_checkpoint(best_model_path)
                val_metrics = trainer.test(model, data_module.val_dataloader())[0]
            else:
                print(colored("> Using latest epoch", 'red'))
                val_metrics = trainer.test(model, data_module.val_dataloader())[0]


            preds_and_fc = perform_predictions(model, data_module.val_dataloader())
            file_name = train_make_plot(preds_and_fc.actuals, preds_and_fc.forecasts, save ="_{}_fold_{}_val_{}".format(model_name, fold,j), neptune_logger=None)
            neptune_logger.experiment["train/images"].log(File(file_name))

            neptune_logger.experiment['train/logged_metrics/' +str(j)] = trainer.logged_metrics
            # Interim step for total performance calculation 

            total_performance['mean_val_mae'].append(val_metrics['test_MAE']) # Called test MAE by trainer (val MAE)
            total_performance['mean_val_mape'].append(val_metrics['test_MAPE'])
            total_performance['mean_val_smape'].append(val_metrics['test_SMAPE'])
            total_performance['mean_val_rmse'].append(val_metrics['test_RMSE'])
            total_performance['mean_val_loss'].append(val_metrics['test_loss'])

            new_train_val_data_dict = {
                            "train_indices": new_train_indices, 
                            "val_indices": new_val_indices, 
                            'val_mae'    : mean(total_performance['mean_val_mae']), 
                            'val_rmse'   : mean(total_performance['mean_val_rmse']), 
                            'val_mape'   : mean(total_performance['mean_val_mape']), 
                            'val_smape'  : mean(total_performance['mean_val_smape']),
                            'val_loss'   : mean(total_performance['mean_val_loss'])}
        
            training_and_val_data[j] = new_train_val_data_dict 

        # Calculate standard deviation
        total_performance['sd_val_mae']   =  np.std(total_performance['mean_val_mae'])
        total_performance['sd_val_smape'] =  np.std(total_performance['mean_val_smape'])
        total_performance['sd_val_mape']  =  np.std(total_performance['mean_val_mape'])
        total_performance['sd_val_rmse']  =  np.std(total_performance['mean_val_rmse'])
        total_performance['sd_val_loss']  =  np.std(total_performance['mean_val_loss'])


        #  Performance result :Calulate mean of all fold performance
        total_performance['mean_val_mae']    = mean(total_performance['mean_val_mae'] )
        total_performance['mean_val_mape']   = mean(total_performance['mean_val_mape'] )
        total_performance['mean_val_smape']  = mean(total_performance['mean_val_smape'] )
        total_performance['mean_val_rmse']   = mean(total_performance['mean_val_rmse'] )
        total_performance['mean_val_loss']   = mean(total_performance['mean_val_loss'] )


        return total_performance, training_and_val_data



def non_overlapping_train_val_folds(index, prediction_length, train_indices, val_indices, test_indices):
            """
            Helper function for setting rolling non overlapping validation indicies.

            Ignores test indices by always returning the same test indices, which are not used for model selection. 
            """
            end_val_index = index*prediction_length # For validation sets
            new_train_indices = train_indices + val_indices[0:(index-1)*prediction_length]
            new_val_indices = val_indices[(index-1)*prediction_length:end_val_index]
            new_test_indices = test_indices
            return new_train_indices, new_val_indices, new_test_indices
            



def overlapping_train_val_folds_old(index, prediction_length, train_indices, val_indices, test_indices):
            """
            NB!

            Assumes that val_length == prediction_length

            Helper function for setting rolling non overlapping validation indicies.

            Ignores test indices by always returning the same test indices, which are not used for model selection. 
            """
            starting_index = index*prediction_length # For test sets
            new_train_indices = train_indices + val_indices + test_indices[0:(index-1)*prediction_length]
            val_and_test_indices = val_indices + test_indices
            new_val_indices = val_and_test_indices[(index-1)*prediction_length:starting_index-1]
            new_test_indices = test_indices[starting_index:starting_index+prediction_length] 
            return new_train_indices, new_val_indices, new_test_indices


def overlapping_train_val_folds(index, prediction_length, train_indices, val_indices, test_indices):
            """
            NB!
            

            Helper function for setting rolling non overlapping validation indicies.

            Ignores test indices by always returning the same test indices, which are not used for model selection. 
            """
            val_size = len(val_indices)
            starting_index =  (index-1)*prediction_length # For test sets
            new_train_indices = train_indices + val_indices + test_indices[0:(index-1)*prediction_length]
            val_and_test_indices = val_indices + test_indices
            #new_val_indices = val_and_test_indices[(index-1)*val_size:(index-1)*val_size + val_size + (starting_index-1)]
            new_val_indices = val_and_test_indices[(index-1)*prediction_length: val_size + (index-1)*prediction_length]
            new_test_indices = test_indices[starting_index:starting_index+prediction_length] 
            return new_train_indices, new_val_indices, new_test_indices
            

def cross_validation(df: pd.DataFrame, 
                     model: BaseModelWithCovariates,    
                     cv,
                     model_name:str,
                     timeseries_kwargs: dict,
                     model_configuration: dict,            
                     verbose = False,
                     gpu_bool = gpus) -> tuple:
    """
    Function that cross validates a given model configuration and returns the mean of a set of performance
    metrics
    """
    start_cv = time.time()

    # Neptune logger initialisations
    if model_configuration.get("neptune_logger") is False:
            neptune_logging = False
    else: neptune_logging = True
    if neptune_logging:
        neptune_logger = neptune.init(
            project         = "MasterThesis/Price",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY3M2E4ZC0xYjIzLTQ5OWYtYTA0MC04NmVjOWRkZWViNmIifQ==",
            name = model_name,
            description = "Cross validated using" + cv.get_type() + " with " + str(cv.n_splits) + " splits",
            source_files=["model_definitions.py", "run_lstm.py", "run_tft.py", "run_deepar.py", "run_tft.py" "run_gru.py"]
        )  
        neptune_logger['Type']                                  = "Cross validated run"
        neptune_logger['Target']                                = timeseries_kwargs.get('target')
        neptune_logger['Architecture']                          = model_name
        neptune_logger['model/configuration']                   = model_configuration
    else: neptune_logger = None




    try:
        neptune_logger['model/hparams'] = model.hparams
    except:
        print(">No hparams")

    
    # Initial values and dictionary structures
    total_performance = {
        'mean_val_mae'  : [],
        'mean_val_mape' : [],
        'mean_val_smape' : [],
        'mean_val_rmse' : [],
        'mean_val_loss' : []
    }
    training_and_val_data    = {}

    # Specifify prediction length
    prediction_length = timeseries_kwargs.get('max_prediction_length')
    print(colored("PRED LENGTH{}".format(prediction_length), "green"))

    # Initial increment value
    i = 0
    print(colored(">" + cv.get_type() + " cross validation pending... ", "green"),  colored("[", "green"), colored(i, "green"), colored("out of", "green"), colored(cv.get_n_splits(), "green"), colored("]", "green"))
    # Iterate through cross validation nested-list
    kwargs = {'model_name': model_name}
    for train_indices, val_indices, test_indices in cv.split(df):

        # Start timer per iteration session
        start_iter = time.time()
        print(colored(">>>>> Fold number:" + str(i), 'green'))
        
        # Record performance per fold
        fold_performance = {
        'mean_val_mae'  : [],
        'mean_val_mape' : [],
        'mean_val_smape' : [],
        'mean_val_rmse' : [],
        'mean_val_loss' : []
        }

        complete_folds =  math.floor(len(val_indices)/prediction_length) # Complete validation folds
        print(colored("> Starting walk-forward forecasts on validation set", "cyan"))
        for j in range(1, complete_folds+1):
            
            new_train_indices, new_val_indices, new_test_indices = non_overlapping_train_val_folds(j, prediction_length, train_indices, val_indices, test_indices)
            data_module = DataModuleTimeSeries(df = df, train_indices = new_train_indices, val_indices=new_val_indices, test_indices=new_test_indices, **timeseries_kwargs)
            if verbose: print(colored(">  Val indices {}".format(new_val_indices) ,'cyan'))
            
            trainer, model = train_model(data_module=data_module, model= model, neptune_logger = None, gpu_bool=gpu_bool,model_configuration=model_configuration, **kwargs)
            
            
            if neptune_logging:
                neptune_logger['train/loss'] = trainer.logged_metrics
         
            try: 
                model_call = model_configuration.get("model_call")
                
            except:
                    raise ValueError("Missing stored")
            

            # Perform ensembles, load from best checkpoint, or take last epoch 
            if model_configuration.get("k_best_checkpoints") is not None:
                print(colored("> Model call {}".format(model_call), 'green'))
                print(colored(" > Using best k best", 'green'))
                actuals_and_fc = ensemble_checkpoints(trainer, model_call, data_module.val_dataloader())

            elif model_configuration.get('use_best_checkpoint'): 
 
                print(colored("> Using best checkpoint....", 'green'))
                best_model_path = trainer.checkpoint_callback.best_model_path
                model = model_call.load_from_checkpoint(best_model_path)
                actuals_and_fc = perform_predictions(model, data_module.val_dataloader())
                #val_metrics = trainer.test(model, data_module.val_dataloader())[0]
            else:

                actuals_and_fc = perform_predictions(model, data_module.val_dataloader())
            

            print("> Preds and FC: {}".format(actuals_and_fc))
            #Calculate metrics
            val_metrics = calc_metrics(actuals_and_fc)

            # Plot predictions
            train_make_plot(actuals_and_fc.actuals, actuals_and_fc.forecasts, save ="_{}_fold_{}_val_{}".format(model_name, i,j), legend_text= "MAE: {:.3f}".format(val_metrics['test_MAE']), neptune_logger=neptune_logger)
                

            # Interim step for total performance calculation 
            total_performance['mean_val_mae'].append(val_metrics['test_MAE']) # Called test MAE by trainer (val MAE)
            total_performance['mean_val_mape'].append(val_metrics['test_MAPE'])
            total_performance['mean_val_smape'].append(val_metrics['test_SMAPE'])
            total_performance['mean_val_rmse'].append(val_metrics['test_RMSE'])
            #total_performance['mean_val_loss'].append(val_metrics['test_loss'])


            fold_performance['mean_val_mae'].append(val_metrics['test_MAE']) # Called test MAE by trainer (val MAE)
            fold_performance['mean_val_mape'].append(val_metrics['test_MAPE'])
            fold_performance['mean_val_smape'].append(val_metrics['test_SMAPE'])
            fold_performance['mean_val_rmse'].append(val_metrics['test_RMSE'])
            #fold_performance['mean_val_loss'].append(val_metrics['test_loss'])

            print(colored(">>> End of walkforward prediction {} of {} val metrics {} ".format(j, complete_folds, val_metrics), "green"))
            print(colored(">>> End of walkforward averge val mae so far {} ".format(mean(total_performance['mean_val_mae'])), "green"))
            #### END OF WALKFORWARD PREDICTION ###

        i += 1  # Index increment 


        new_train_val_data_dict = {
                        "train_indices": new_train_indices, 
                        "val_indices": new_val_indices, 
                        'val_mae'    : mean(fold_performance['mean_val_mae']), 
                        'val_rmse'   : mean(fold_performance['mean_val_rmse']), 
                        'val_mape'   : mean(fold_performance['mean_val_mape']), 
                        'val_smape'  : mean(fold_performance['mean_val_smape'])}
    
        training_and_val_data[i] = new_train_val_data_dict 

        end_iter = time.time() # End timer for iteration

        iter_time = end_iter - start_iter
        print(colored(">>> Iteration time used:", "red"), colored(str(round(iter_time,1)), "red"), colored("seconds", "red"))

        # Deleting and clearing gpu memory
        if gpu_bool == 1:    
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        # Deleting and clearing gpu memory
        gc.collect()
    # Calculate standard deviation
    total_performance['sd_val_mae']   =  np.std(total_performance['mean_val_mae'])
    total_performance['sd_val_smape'] =  np.std(total_performance['mean_val_smape'])
    total_performance['sd_val_mape']  =  np.std(total_performance['mean_val_mape'])
    total_performance['sd_val_rmse']  =  np.std(total_performance['mean_val_rmse'])
    #total_performance['sd_val_loss']  =  np.std(total_performance['mean_val_loss'])
    
    
    #  Performance result :Calulate mean of all fold performance
    total_performance['mean_val_mae']    = mean(total_performance['mean_val_mae'] )
    total_performance['mean_val_mape']   = mean(total_performance['mean_val_mape'] )
    total_performance['mean_val_smape']  = mean(total_performance['mean_val_smape'] )
    total_performance['mean_val_rmse']   = mean(total_performance['mean_val_rmse'] )
    #total_performance['mean_val_loss']   = mean(total_performance['mean_val_loss'] )



    end_cv = time.time() # End time

    cv_time = end_cv - start_cv  # CV time used
    
   
    
    # Console message and results
    print(colored("\n"+ ">"+ cv.get_type()+ "cross Validation complete...", "blue"))
    print(colored(">>> Total time used:", "red"), colored(str(round(cv_time,1)), "red"), colored("seconds", "red"))
    print(colored(">>> Mean Performance Result:", "green"), colored(total_performance, "green"), colored("<<<", "green"))

    if neptune_logging:
        neptune_logger['cv/time'] = cv_time
        neptune_logger['model/total_performance']     = total_performance
        neptune_logger['model/training_and_val_data'] = training_and_val_data
        neptune_logger.stop()



    manual_test_logs({'total_performance':total_performance, 'training and validation data': training_and_val_data}, model_name)
    return total_performance, training_and_val_data


def ensemble_checkpoints(trainer, model_call, dataloader: DataLoader) -> pd.DataFrame:
    """
    Perform checkpoint ensembling for top_k checkpoints recorded
    in trainer object. 
    (by averaging predictions)
    """
    
    top_model_checkpoint_paths = trainer.checkpoint_callback.best_k_models.keys()
    n_models  = len(top_model_checkpoint_paths)
    actuals_and_fc = pd.DataFrame()
    for model_path in top_model_checkpoint_paths:
        model = model_call.load_from_checkpoint(model_path)
        new_preds_and_fc = perform_predictions(model, dataloader)
        #print(" >New preds {}".format(new_preds_and_fc))
        try:
            actuals_and_fc['forecasts'] = actuals_and_fc['forecasts'] +new_preds_and_fc['forecasts']
            
        except:
            #print(" > First pass ensemble checkpoints")
            actuals_and_fc['forecasts'] = new_preds_and_fc['forecasts']
            actuals_and_fc['actuals'] = new_preds_and_fc['actuals']
    actuals_and_fc.forecasts = actuals_and_fc.forecasts/n_models
    return actuals_and_fc



def calc_metrics(actuals_and_fc : pd.DataFrame) -> dict:
    """
    Calulates test/val metrics for dataframe of forecasts and actuals
    """
    #print(" > Actuals and forecasts {}".format(actuals_and_fc))
    test_mae   = mean_absolute_error(actuals_and_fc.actuals, actuals_and_fc.forecasts)
    test_mape  = mean_absolute_percentage_error(actuals_and_fc.actuals, actuals_and_fc.forecasts)
    test_smape = smape(actuals_and_fc.actuals, actuals_and_fc.forecasts)
    test_rmse  = mean_squared_error(actuals_and_fc.actuals, actuals_and_fc.forecasts, squared = False)
    metrics = {'test_MAE': test_mae, 'test_MAPE':test_mape,  'test_SMAPE': test_smape, 'test_RMSE':test_rmse}
    return metrics


#ensemble_checkpoints(trainer, DeepAR)


#################################################################################################################################
##### Generalization estimation  #########################################################################################
#################################################################################################################################
def cross_validation_test_folds(df: pd.DataFrame, 
                     model: BaseModelWithCovariates,    
                     cv,
                     model_name:str,
                     timeseries_kwargs: dict,
                     model_configuration: dict,            
                     verbose = False,
                     gpu_bool = gpus) -> tuple:
    """
    Function that cross validates a given model configuration and returns the mean of a set of performance
    metrics
    """
    start_cv = time.time()

      
    # Checking model configuration

    neptune_logger = neptune.init(
        project         = "MasterThesis/Price",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY3M2E4ZC0xYjIzLTQ5OWYtYTA0MC04NmVjOWRkZWViNmIifQ==",
        name = model_name,
        description = "Cross validated using" + cv.get_type() + " with " + str(cv.n_splits) + " splits",
        source_files=["model_definitions.py", "model_implementations.py"]
    )  

    neptune_logger['Type']                                              = "Cross validated run"
    neptune_logger['Target']                                            = timeseries_kwargs.get('target')
    neptune_logger['Architecture']                                      = model_name
    neptune_logger['model/configuration'] = model_configuration

    try:
        neptune_logger['model/hparams'] = model.hparams_initial
    except:
        print(">No hparams")

    
    # Initial values and dictionary structures
    #evaluation_dict   = {}
    total_performance = {
        'mean_test_mae'  : [],
        'mean_test_mape' : [],
        'mean_test_rmse' : [],
        'mean_test_loss' : []
    }
    training_and_val_data    = {}

    # Specifify prediction length
    prediction_length = timeseries_kwargs.get('max_prediction_length')


    # Initial increment value
    i = 0
    print(colored(">" + cv.get_type() + " cross validation pending... ", "green"),  colored("[", "green"), colored(i, "green"), colored("out of", "green"), colored(cv.get_n_splits(), "green"), colored("]", "green"))
    # Iterate through cross validation nested-list
    
    for train_indices, val_indices, test_indices in cv.split(df):

        # Start timer per iteration session
        start_iter = time.time()
        print(colored(">>>>> Fold number:" + str(i), 'green'))
        
        if verbose: print(colored("> TESTING: Val and test indices" + " val indices" + str(val_indices) + " test indices" + str(test_indices), 'cyan'))
        
        kwargs = {'model_name': model_name}

        complete_folds =  math.floor(len(test_indices)/prediction_length)
        print(colored("> Starting walk-forward forecasts on test set", "cyan"))
        for j in range(1, complete_folds+1):
            
            new_train_indices, new_val_indices, new_test_indices = overlapping_train_val_folds(j, prediction_length, train_indices, val_indices, test_indices)
            data_module = DataModuleTimeSeries(df = df, train_indices = new_train_indices, val_indices=new_val_indices, test_indices=new_test_indices, **timeseries_kwargs)
            
            #if verbose: print("TRAIN:", data_module.train_dataloader().dataset.data, '\n', "VALIDATION:", data_module.val_dataloader().dataset.data)
            
            trainer, model = train_model(data_module=data_module, model= model, neptune_logger = None, gpu_bool=gpu_bool,model_configuration=model_configuration, **kwargs)

            # Record test score 
            if verbose: print(colored("> Appending test results...", "blue"))
        
            test_metrics = trainer.test(model, data_module.test_dataloader())[0]
            if verbose: print(colored("> Test metrics {}".format(test_metrics, "blue")))
        

            # Record train and validation data


            """            new_train_val_data_dict = {"train_indices": new_train_indices, 
                            "val_indices": new_val_indices, 'test_indices':new_test_indices}
                        new_train_val_data_dict.update(trainer.logged_metrics)
                        new_train_val_data_dict.update(test_metrics)
                        training_and_val_data[i] = new_train_val_data_dict
            """

            # Interim step for total performance calculation 
            total_performance['mean_test_mae'].append(test_metrics['test_MAE'])
            total_performance['mean_test_mape'].append(test_metrics['test_MAPE'])
            total_performance['mean_test_rmse'].append(test_metrics['test_RMSE'])
            total_performance['mean_test_loss'].append(test_metrics['test_loss'])

            print(colored(">>> End of walkforward prediction {} of {} validation metrics {} ".format(j, complete_folds, test_metrics), "green"))
            print(colored(">>> End of walkforward averge test mae so far {} ".format(mean(total_performance['mean_test_mae'])), "green"))
            #### END OF WALKFORWARD PREDICTION ###

        i += 1  # Index increment 



        
        end_iter = time.time() # End timer for iteration

        iter_time = end_iter - start_iter
        print(colored(">>> Iteration time used:", "red"), colored(str(round(iter_time,1)), "red"), colored("seconds", "red"))

        # Deleting and clearing gpu memory
        if gpu_bool == 1:    
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
    
    #  Performance result :Calulate mean of all fold performance
    total_performance['mean_test_mae']   = mean(total_performance['mean_test_mae'] )
    total_performance['mean_test_mape']  = mean(total_performance['mean_test_mape'] )
    total_performance['mean_test_rmse']  = mean(total_performance['mean_test_rmse'] )
    total_performance['mean_test_loss']  = mean(total_performance['mean_test_loss'] )


    end_cv = time.time() # End time

    cv_time = end_cv - start_cv  # CV time used
    
    neptune_logger['cv/time'] = cv_time
    
    # Console message and results
    print(colored("\n"+ ">"+ cv.get_type()+ "cross Validation complete...", "blue"))
    print(colored(">>> Total time used:", "red"), colored(str(round(cv_time,1)), "red"), colored("seconds", "red"))
    print(colored(">>> Mean Performance Result:", "green"), colored(total_performance, "green"), colored("<<<", "green"))

    neptune_logger['model/total_performance'] = total_performance 
    neptune_logger['model/training_and_val_data'] = training_and_val_data
    neptune_logger.stop()

    manual_test_logs({'total_performance':total_performance, 'training and validation data': training_and_val_data}, model_name)
    return total_performance, training_and_val_data



def find_initial_learning_rate(model, data_module, model_configuration):
    """
    Plot and returns a suggested learning rate
    """

    
    gradient_clip_val = model_configuration.get('gradient_clip_val')
    if gradient_clip_val is None:
        gradient_clip_val = 0.1
    stoch_weight_avg = StochasticWeightAveraging(swa_epoch_start = 5, swa_lrs=1e-2)
    trainer = flash.Trainer(
        max_epochs=50,
        gpus=int(torch.cuda.is_available()),
        gradient_clip_val= gradient_clip_val,
        callbacks = [stoch_weight_avg]
    )

    res = trainer.tuner.lr_find(model, datamodule=data_module, min_lr=1e-9)
    print(f"Suggested learning rate: {res.suggestion()}")
    res.plot(show=True, suggest=True).show()
    return res

########################  



def manual_test_logs(configurations : dict, model_name:str):

    """
    Manual test logs to txt files
    """
    
    list_of_files = glob.glob('manual_logs/*') # * means all if need specific format then *.csv
    try:
        last_model_file = max(list_of_files, key=os.path.getctime)
        model_number = str(int(last_model_file.split("/")[1].split("_")[1].split(".")[0]) + 1)
        print("> Trying model number", model_number)
    except:
        model_number = "1"
    # Create new file
    if not os.path.exists("manual_logs"):
            os.makedirs("manual_logs")
    with open('manual_logs/' + model_name  + '_' + model_number + '.txt', 'w') as f:
        f.writelines('Model number:' + model_number)
        
        for k, v in configurations.items():
            f.writelines('\n')
            f.writelines(k+': '+ str(v))

def save_configurations(model_config:dict, timeseries_kwargs:dict, model_name : str):
    """
    """
    # Create new file
    if not os.path.exists("manual_logs"):
            os.makedirs("manual_logs")
    if not os.path.exists("manual_logs/configs"):
            os.makedirs("manual_logs/configs")
    try:
        list_of_files = glob.glob('manual_logs/configs/*') # * means all if need specific format then *.csv
        last_model_file = max(list_of_files, key=os.path.getctime)
        model_number = str(int(last_model_file.split("/")[2].split("_")[1].split(".")[0]) + 1)
        print("> Trying model number", model_number)
    except:
        model_number = "1"

    with open('manual_logs/configs/' + model_name  + '_' + model_number + '.txt', 'w') as f:
        f.writelines('Model name:' + model_name)
        f.writelines('\n')
        f.writelines('Model number:' + model_number)
        f.writelines('\n')
        f.writelines("model_config" + str(model_config))
        f.writelines('\n')
        f.writelines("timeserieskwargs" + str(timeseries_kwargs))




#################################################################################################################################
###################### Predictions and plots       ##############################################################################
#################################################################################################################################

def smape(a, f):
    """
    Implementation of Symmetric Mean Absolute Percentage Error.
    For use in benchmark models mostly
    """
    try:
        a = a.to_numpy()
        f = f.to_numpy()
        return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f)))
    except:
        return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f)))




def perform_predictions(model : BaseModel, test: Optional[DataLoader], starting_index: int = 0) -> pd.DataFrame:
    """
    Performs predictions using a model and test dataframe and 
    returns a dataframe of predicted and observed data (actual)
    @model: model, not trainer object
    """

    try:
        preds = model.predict(test, return_index = 0, mode = "prediction")
        preds_numpy = preds[0].cpu().detach().numpy().flatten()
        horizon = len(preds_numpy) # Prediction horizon
        #preds_numpy = preds_numpy[-38:-1]
        actuals_numpy = test.dataset.data.get('target')[0][-horizon:]
        preds_and_fc = pd.DataFrame({'actuals': actuals_numpy, 'forecasts':preds_numpy})
        return preds_and_fc
    except:
        # Assumes dataframe
        preds = model.predict(test, return_index = 0, mode = "prediction")
        preds_numpy = preds[0].cpu().detach().numpy().flatten()
        horizon = len(preds_numpy) # Prediction horizon
        preds_numpy = preds_numpy[starting_index:horizon+1]
        actuals_numpy = test.dataset.data.get('target')[0].numpy()[starting_index:horizon]
        preds_and_fc = pd.DataFrame({'actuals': actuals_numpy, 'forecasts':preds_numpy})
        return preds_and_fc



def perform_rolling_predictions(model, test, prediction_length:int,  starting_index = 0):
    """
    Performs predictions for the entire length of the input test dataframe
    'Rolls' starting_index forward til end
    """
    complete_folds = int(np.floor(len(test)/prediction_length))
    starting_index = 0
    preds_and_actuals = pd.DataFrame()
    for fold in range(0, complete_folds):
        predictions = perform_predictions(model, 'intraday_price_difference', test, starting_index)
        preds_and_actuals = pd.concat([preds_and_actuals, predictions])
        starting_index += prediction_length
    return preds_and_actuals




def plot_predictions(preds_and_actual: pd.DataFrame, plot_title) -> None:
    """
    Wrapper function around analysis.sns_lineplot
    Transforms the input data series to long format
    """
    long_preds_and_actual = preds_and_actual.melt(id_vars = ['datetime'], 
                var_name = 'series_type', 
                value_name = 'value', value_vars = ['preds', 'actuals'])
    sns_lineplot(long_preds_and_actual, x = 'datetime', y = 'value',  hue='series_type', plot_title= plot_title)


def plot_predictions_simple(model, test):

    """ 
    Plots in-sample predictions
    Uses built-in plot_predictions of model
    """
    raw_predictions, x = model.predict(test, 
                                      mode = "raw", 
                                      return_x = True)

    model.plot_prediction(x, 
                             raw_predictions, 
                             idx               = 0, 
                             add_loss_to_title = True)

    return raw_predictions, x


def variable_importance(model: RecurrentNetwork, test_set: pd.DataFrame) -> dict:
    """
    Wrapper functionn for variable importance 
    @test_set : dataloader or dataframe
    """
    raw_predictions, _ = model.predict(test_set, mode="raw", return_x=True)
    interpretations = model.interpret_output(raw_predictions, reduction="sum")
    
    model.plot_interpretation(interpretations) # Plot interpretations

    return interpretations



                    

#################################################################################################################################
######################################################## Grid search methods ####################################################
#################################################################################################################################




def grid_search_extended(df : pd.DataFrame, model_call, timeseries_kwargs:dict, model_configurations: Dict[str, list], verbose = False):
    """

    Performs a grid search by training multiple model according
    to the model configurations specified. 

    Configurations id
    @model_call: function passed as parameter
    """
    
    keys, values = zip(*model_configurations.items())
    model_configurations_list = [dict(zip(keys, v)) for v in itertools.product(*values)] # All possible model configurations
    print(colored("> Starting grid search. Number of configurations; " + str(len(model_configurations_list)), "blue"))
    print(colored("> All models" + str(model_configurations_list), "blue"))
    for model_configuration in model_configurations_list:
            print("Model config", model_configuration)
            print(colored("> Starting new model using config : " + str(model_configuration), 'green'))
            model_call(df, model_configuration, timeseries_kwargs, verbose = verbose)

