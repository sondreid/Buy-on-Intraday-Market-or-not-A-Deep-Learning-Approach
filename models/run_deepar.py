
from pytorch_forecasting import DeepAR
from model_definitions import *
from utilities import shift_covariates
from TFTtuning import *
from RNNtuning import *
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
from pytorch_lightning.loggers import NeptuneLogger
from model_definitions import cross_validation
import tensorflow as tf


"""

Script for Deepar implementation as
described in 
https://reader.elsevier.com/reader/sd/pii/S0169207019301888?token=2B6427C9CEEB9BBEB3287713E98F5E9B153021BEA1DC4881B1198ADEDC9D534C594D2269EB52691B62C31C1D429D9D3A&originRegion=eu-west-1&originCreation=20220507184753 


"""




target                                  = 'dayahead_price'

# Specifications for TimeSeriesDataset
lag_vars_expanded = capacity_vars = [covariate for covariate in series_df_shifted.columns if "lag" in covariate]

forecast_vars = [covariate for covariate in series_df_shifted.columns if "Forecast" in covariate or 'REMIT' in covariate]

capacity_vars = [covariate for covariate in series_df_shifted.columns if "Day-Ahead Capacity" in covariate]

time_varying_known_categoricals         = ['day', 'hour', 'year', 'week', 'month', 'nordlink', 'northsealink']
time_varying_unknown_categoricals       = ['covid']

non_forecast_vars = [covariate for covariate in series_df_shifted.columns if covariate not in forecast_vars + capacity_vars + time_varying_known_categoricals + time_varying_unknown_categoricals + ['seq']+['datetime']+['time_groups']]
non_forecast_vars =[covariate for covariate in non_forecast_vars if 'price_lag' not in covariate]
time_varying_known_reals                = forecast_vars + capacity_vars
time_varying_unknown_reals              =  non_forecast_vars + lag_vars





def setup_deepar(df, deepar_model_config, timeseries_kwargs):
    """
    Setting up common parameters for TFT
    """
    

    # Parameters passed to trainer and other parameters
    deepar_model_parameters =   ['accumulate_grad_batches',
                    'max_epochs',
                    'gradient_clip_val',
                    'patience',
                    'min_delta',
                    'auto_scale_batch_size',
                    'use_best_checkpoint',
                    'k_best_checkpoints',
                    'model_call',
                    'print_predictions',
                    'swa_epoch_start',
                    'neptune_logger']

    # Parameters passed to TemporalFusionTransformers                    
    deepar_configuration =  [
        'cell_type',
        'hidden_size',
        'rnn_layers',
        'dropout',
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
        'logging_metrics',
        'reduce_on_plateau_patience',
        'reduce_on_plateau_reduction',
        'reduce_on_plateau_min_lr',
        'optimizer_params',
        'target',
        'optimizer']


    #timeseries_extra_params = ['num_workers', 'batch_size']
    timeseries_kwargs['num_workers'] = deepar_model_config.get('num_workers')
    timeseries_kwargs['batch_size'] = deepar_model_config.get('batch_size')
    filtered_deepar_model_parameters = {key: dict_item for key, dict_item in deepar_model_config.items() if key.startswith(tuple(deepar_model_parameters))}
    print(colored("Filtered parameters {}".format(filtered_deepar_model_parameters, 'green')))
    filtered_deepar_configuration = {key: dict_item for key, dict_item in deepar_model_config.items() if key.startswith(tuple(deepar_configuration))}
    try:
        data_module = DataModuleTimeSeries(df  = df, **timeseries_kwargs)
        deepar_model = DeepAR.from_dataset(
            dataset = data_module.train_dataloader().dataset, 
            **filtered_deepar_configuration)
    except:
        raise ValueError("Missing parameters")
    return data_module, deepar_model, filtered_deepar_model_parameters



def test_deepar(df, deepar_model_config,timeseries_kwargs):
    """
    Wrapper function for running cross validated TFT run
    """
    data_module, deepar_model, filtered_tft_model_parameters = setup_deepar(df, deepar_model_config, timeseries_kwargs)

    trainer, model = train_model(
        data_module = data_module, 
        model       = deepar_model,  
        gpu_bool    = gpus, 
        neptune_logger =  None,
        model_configuration = filtered_tft_model_parameters)
     #interpretations = variable_importance(model, data_module.test_dataloader())
    return trainer, model




def refit_deepar(df :pd.DataFrame, model_name:str, tft_model_config:dict, timeseries_kwargs:dict):
    """
    Wrapper function for running cross validated TFT run
    """
    data_module, tft_model, filtered_tft_model_parameters = setup_deepar(df, tft_model_config, timeseries_kwargs)

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




def run_deepar_model(df : pd.DataFrame, deepar_model_config:dict, timeseries_kwargs:dict, verbose = False):
    """
    Wrapper function for running cross validated TFT run
    """
    _, deepar_model, filtered_tft_model_parameters = setup_deepar(df, deepar_model_config, timeseries_kwargs)
    cross_validation(df = df, model = deepar_model, cv = cv, model_name = "deepar",  timeseries_kwargs = timeseries_kwargs, model_configuration = filtered_tft_model_parameters, verbose = verbose)







if __name__ == "__main__":


    ### deepar ##
    # Notes
    # Don't use load from best_checkpoint if Ctrl+C training
    def run_grid_search():
        
        timeseries_kwargs = {'time_idx':            'seq',
            'target':                              target, 
            'min_encoder_length' :                 1,  
            'max_encoder_length' :                 300,  
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
            'add_encoder_length':                  True,
            'num_workers':                           11,
            'scale':                                False} 

        
        deepar_model_config =   {
                'batch_size':                           [128], 
                'num_workers':                          [11],  
                'max_epochs':                           [70], 
                'patience':                             [16],
                'cell_type':                            ['GRU'],
                'hidden_size':                          [128],
                'rnn_layers':                           [3],
                'dropout':                              [0.4],
                'min_delta':                            [0.05],
                #Multihorizon metrics should not be used
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
                'output_size':                          [1],
                'print_predictions':                    [True],
                #'target_lags':                          {},
                #'logging_metrics':                      None
                'optimizer':                            ['ranger'],
                'gradient_clip_val'                     :[0.6],
                'learning_rate' :                       [0.001],
                'accumulate_grad_batches':              [5], 
                'use_best_checkpoint':                  [False],
                'k_best_checkpoints' :                   [3],
                'reduce_on_plateau_patience':           [3],
                'reduce_on_plateau_reduction' :         [3.0],
                'model_call' :                         [DeepAR],
                'neptune_logger':                       [True],
                'swa_epoch_start' :                    [13] 
        }
        
        grid_search_extended(series_df_shifted, run_deepar_model, timeseries_kwargs, deepar_model_config, verbose=True)
        

    def run_refit():
        # Refitting 
        deepar_model_config =   {
        'cell_type':                       'LSTM',
        'batch_size':                           64, 
        'num_workers':                          11,  
        'max_epochs':                           70, 
        'patience':                             7,
        'hidden_size':                          128,
        'hidden_size':                          32,
        'rnn_layers':                            4,
        #'loss':                                 metrics.MAE(),   
        'dropout':                              0.1,
        #'static_categoricals':                  [],
        #'static_reals':                         [],
        'time_varying_categoricals_encoder':    time_varying_unknown_categoricals + time_varying_known_categoricals,
        'time_varying_categoricals_decoder':    time_varying_unknown_categoricals + time_varying_known_categoricals,
        #'categorical_groups':                   {},
        'time_varying_reals_encoder':           time_varying_unknown_reals + time_varying_known_reals,
        'time_varying_reals_decoder':           [x for x in time_varying_unknown_reals if x != target] + time_varying_known_reals,
        #'embedding_sizes':                      {},
        #'embedding_paddings':                   [],
        #'embedding_labels':                     {},
        #'x_reals':                              [],
        #'x_categoricals':                       [],
        'output_size':                          1,
        #'target_lags':                          {},
        #'loss':                                 metrics.QuantileLoss(), # QuantileLoss does not work with recurrent network
        #'logging_metrics':                      None
        #'n_plotting_samples ' : 100,
        'optimizer':  'ranger',
        'target': target, 
        'accumulate_grad_batches':2, 
        'use_best_checkpoint': False
        }

        trainer, model = refit_deepar(series_df_shifted, 'deepar_refit_1', deepar_model_config, timeseries_kwargs)

    ############# Run either grid search or refit
    #run_refit()
    target                                  = 'dayahead_price'
    run_grid_search()
    target                                  = 'intraday_price'
    run_grid_search()
