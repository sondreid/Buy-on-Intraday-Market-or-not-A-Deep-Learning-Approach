
from pytorch_lightning import LightningDataModule
from model_definitions import *
from utilities import shift_covariates
from TFTtuning import *
from RNNtuning import *
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from pytorch_lightning.loggers import NeptuneLogger
from model_definitions import cross_validation
import tensorflow as tf





target                                  = 'intraday_price'

# Specifications for TimeSeriesDataset
lag_vars_expanded = capacity_vars = [covariate for covariate in series_df_shifted.columns if "lag" in covariate]
# Specifications for TimeSeriesDataset


forecast_vars = [covariate for covariate in series_df_shifted.columns if "Forecast" in covariate or 'REMIT' in covariate]

capacity_vars = [covariate for covariate in series_df_shifted.columns if "Day-Ahead Capacity" in covariate]

time_varying_known_reals                = forecast_vars + capacity_vars
time_varying_unknown_reals              =  non_forecast_vars + lag_vars_expanded






def setup_lstm(df, lstm_model_config, timeseries_kwargs):
    """
    Setting up common parameters for TFT
    """
    

    # Parameters passed to trainer and other parameters
    lstm_model_parameters =   [
                    'accumulate_grad_batches',
                    'max_epochs',
                    'gradient_clip_val',
                    'patience',
                    'min_delta',
                    'auto_scale_batch_size',
                    'use_best_checkpoint',
                    'k_best_checkpoints',
                    'model_call',
                    'swa_epoch_start',
                    'print_predictions',
                    'neptune_logger']

    # Parameters passed to TemporalFusionTransformers                    
    lstm_configuration =  [
        'cell_type',
        'hidden_size',
        'rnn_layers', 
        'dropout',
        'output_size',
        'loss',
        'time_varying_categoricals_encoder',
        'time_varying_categoricals_decoder',
        'categorical_groups',
        'time_varying_reals_encoder',
        'time_varying_reals_decoder',
        'x_reals',
        'x_categoricals',
        'embedding_sizes',
        'embedding_paddings',
        'embedding_labels',
        'learning_rate',
        'log_interval',
        'log_val_interval',
        'log_gradient_flow',
        'reduce_on_plateau_patience',
        'monotone_constaints',
        'share_single_variable_networks',
        'logging_metrics',
        'target',
        'reduce_on_plateau_patience',
        'reduce_on_plateau_reduction',
        'reduce_on_plateau_min_lr',
        'optimizer_params',
        'optimizer']


    #timeseries_extra_params = ['num_workers', 'batch_size']
    timeseries_kwargs['num_workers'] = lstm_model_config.get('num_workers')
    timeseries_kwargs['batch_size'] = lstm_model_config.get('batch_size')
    filtered_lstm_model_parameters = {key: dict_item for key, dict_item in lstm_model_config.items() if key.startswith(tuple(lstm_model_parameters))}
    print(colored("Filtered parameters {}".format(filtered_lstm_model_parameters, 'green')))
    filtered_lstm_configuration = {key: dict_item for key, dict_item in lstm_model_config.items() if key.startswith(tuple(lstm_configuration))}
    try:
        data_module = DataModuleTimeSeries(df  = df, **timeseries_kwargs)
        lstm_model = RecurrentNetwork.from_dataset(
            dataset = data_module.train_dataloader().dataset, 
            **filtered_lstm_configuration)
    except:
        raise ValueError("Missing parameters")
    return data_module, lstm_model, filtered_lstm_model_parameters



def test_lstm(train_df: pd.DataFrame, val_df : pd.DataFrame,  lstm_model_config,timeseries_kwargs):
    """
    Wrapper function for running cross validated TFT run
    """
    data_module, lstm_model, filtered_tft_model_parameters = setup_lstm(train_df, lstm_model_config, timeseries_kwargs)

    data_module.set_train_dataloader(train_df)
    data_module.set_train_dataloader(val_df)
    trainer, model = train_model(
        data_module = data_module, 
        model       = lstm_model,  
        gpu_bool    = gpus, 
        neptune_logger = None,
        model_configuration = filtered_tft_model_parameters)

     #interpretations = variable_importance(model, data_module.test_dataloader())
    return trainer, model




#find_initial_learning_rate()



def refit_lstm(df :pd.DataFrame, model_name:str, tft_model_config:dict, timeseries_kwargs:dict):
    """
    Wrapper function for running cross validated TFT run
    """
    data_module, tft_model, filtered_tft_model_parameters = setup_lstm(df, tft_model_config, timeseries_kwargs)

    neptune_logger = neptune.init(
        project         = "MasterThesis/Price",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkNGY3M2E4ZC0xYjIzLTQ5OWYtYTA0MC04NmVjOWRkZWViNmIifQ==",
        name = model_name,
        description = model_name, 
        source_files=["model_definitions.py", "model_implementations.py"]
    )  

    neptune_logger['Type']                                              = "Refit run"
    neptune_logger['Target']                                            = timeseries_kwargs.get('target')
    neptune_logger['Architecture']                                      = model_name
    neptune_logger['model/configuration/tft_model_config'] = tft_model_config
    neptune_logger['model/configuration/timeseries_kwargs'] = timeseries_kwargs
    checkpoint_callback = ModelCheckpoint(monitor = 'val_loss', mode = 'min')
    neptune_logger['model/configuration']                               = tft_model_config
    kwargs = {'model_name': 'refit_tft', 'model_checkpoint':checkpoint_callback}
    trainer, model = train_model(
        data_module = data_module, 
        model       = tft_model,  
        gpu_bool    = gpus, 
        logger = neptune_logger,
        model_configuration = filtered_tft_model_parameters,
        **kwargs)

    neptune_logger.stop()
    # Manual logging
    print(colored("Saving model", 'cyan'))
    save_configurations(tft_model_config, timeseries_kwargs, model_name)
    if not os.path.exists("refit_runs"):
        os.makedirs("refit_runs")
    torch.save(model.state_dict(), 'refit_runs/'+model_name+'_model_weights.pth')
    #torch.save(model, 'refit_runs/'+model_name+'_model.pth')
    trainer.save_checkpoint('refit_runs/'+ model_name+ '_trainer.cpkt')
    return trainer, model




def run_lstm_model(df : pd.DataFrame, lstm_model_config:dict, timeseries_kwargs:dict, verbose = False):
    """
    Wrapper function for running cross validated LSTM run
    """
    _, lstm_model, filtered_lstm_model_parameters = setup_lstm(df, lstm_model_config, timeseries_kwargs)
    cross_validation(df = df, model = lstm_model, cv = cv, model_name = "LSTM",  timeseries_kwargs = timeseries_kwargs, model_configuration = filtered_lstm_model_parameters, verbose = verbose)




def main():

    def run_grid_search():



        timeseries_kwargs = {
                'time_idx':                            'seq',
                'target':                              target, 
                'min_encoder_length' :                 1,  
                'max_encoder_length' :                 336,  
                'group_ids':                           ['time_groups'],
                'min_prediction_length':               max_prediction_length,
                'max_prediction_length':               max_prediction_length+2,
                'time_varying_known_reals' :           [],
                'time_varying_unknown_reals':          time_varying_unknown_reals + time_varying_known_reals,
                'time_varying_known_categoricals':     [],
                'time_varying_unknown_categoricals':   time_varying_unknown_categoricals + time_varying_known_categoricals, 
                'categorical_encoders':                {'week':         NaNLabelEncoder(add_nan = True), 
                                                        'hour':         NaNLabelEncoder(add_nan = True), 
                                                        'day':         NaNLabelEncoder(add_nan = True), 
                                                        'month':        NaNLabelEncoder(add_nan = True), 
                                                        'year':        NaNLabelEncoder(add_nan = True), 
                                                        'covid':        NaNLabelEncoder(add_nan = True), 
                                                        'nordlink':     NaNLabelEncoder(add_nan = True), 
                                                        'northsealink': NaNLabelEncoder(add_nan = True)
                                                        },
                'target_normalizer' :                  TorchNormalizer(method = 'robust', center = True),
                'scalers': {var: RobustScaler() for var in time_varying_unknown_reals + time_varying_known_reals if var not in target},
                'batch_size':                           128,
                'num_workers':                           5,
                'scale':                                False} 






        lstm_model_config =   {
                'batch_size':                           [128], 
                'num_workers':                          [6],  
                'max_epochs':                           [70], 
                'patience':                             [12],
                'cell_type':                            ['LSTM'],
                'hidden_size':                          [64],
                'rnn_layers':                           [4],
                'loss':                                 [metrics.MAE()],   
                'dropout':                               [0.2],
                'min_delta':                             [0.05],
                #'static_categoricals':                  [],
                #'static_reals':                         [],
                'time_varying_categoricals_encoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
                'time_varying_categoricals_decoder':    [time_varying_unknown_categoricals + time_varying_known_categoricals],
                #'categorical_groups':                   {},
                'time_varying_reals_encoder':           [time_varying_unknown_reals + time_varying_known_reals],
                'time_varying_reals_decoder':           [[x for x in time_varying_unknown_reals if x != target] + time_varying_known_reals],
                #'embedding_sizes':                      {},
                #'embedding_paddings':                   [],
                #'embedding_labels':                     {},
                #'x_reals':                              [],
                #'x_categoricals':                       [],
                'output_size':                           [1],
                'print_predictions':                    [True],
                'target':                                [target],
                'gradient_clip_val'                     : [0.5],
                #'target_lags':                          {},
                #'logging_metrics':                      None
                'optimizer':                            ['ranger'],
                'learning_rate' :                       [0.001],
                'accumulate_grad_batches':              [2], 
                'use_best_checkpoint':                  [False], 
                'k_best_checkpoints' :                   [3],
                'reduce_on_plateau_patience':           [2],
                'reduce_on_plateau_reduction' :         [3.0],
                'model_call' :                          [RecurrentNetwork],
                'neptune_logger':                       [True],
                'swa_epoch_start':                      [15]
        }

        grid_search_extended(series_df_shifted, run_lstm_model, timeseries_kwargs, lstm_model_config, verbose=True)


    run_grid_search()




if __name__ == "__main__":

    main()

    ### LSTM ##

